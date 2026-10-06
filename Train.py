
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_FILE = "retail_pricing_demand_100k.csv"
TEST_FRAC = 0.20
RANDOM_STATE = 42

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
# pandas reads the text "None" (= no promotion) as a missing value, so put the word back
df["promotion_type"] = df["promotion_type"].fillna("None")

# One "series" = one product sold in one region (exactly one row per day)
df["series"] = df["product_id"] + "|" + df["region"]
df = df.sort_values(["series", "date"]).reset_index(drop=True)
print(f"rows: {len(df)} | products: {df['product_id'].nunique()} | series: {df['series'].nunique()} "
      f"| {df['date'].min().date()} to {df['date'].max().date()}")

df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["price_ratio"] = df["current_price"] / df["base_price"]      # 1.0 = full price, 0.8 = 20% off
df["discount_depth"] = df["base_price"] - df["current_price"]

# Recent demand of the same series. shift(1) means today's sales are never used for today.
sales = df.groupby("series")["units_sold"]
df["lag1"] = sales.shift(1)
df["roll7"] = sales.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
df["roll14"] = sales.transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())

# Text columns -> integer codes (trees only need a label, not an order)
CAT_COLS = ["category", "brand", "region", "channel", "season", "promotion_type"]
maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CAT_COLS}
for c in CAT_COLS:
    df[c + "_code"] = df[c].map(maps[c])

FEATURES = (["base_price", "current_price", "discount_depth", "price_ratio", "discount_pct",
             "inventory_level", "day_of_week", "month", "is_weekend", "roll7", "roll14"]
            + [c + "_code" for c in CAT_COLS])
CAT_MASK = [f.endswith("_code") for f in FEATURES]
COL = {f: i for i, f in enumerate(FEATURES)}      # feature name -> column number


dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * (1 - TEST_FRAC))]
train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff]
print(f"train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows) | "
      f"test: {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")

# The promotion label depends on the discount, so when the optimiser changes the price
# it must also change the label. Use the most common label seen at each discount level.
promo_by_discount = train.groupby("discount_pct")["promotion_type"].agg(lambda s: s.mode().iat[0])
promo_levels = promo_by_discount.index.to_numpy(dtype=float)
promo_codes = np.array([maps["promotion_type"][p] for p in promo_by_discount])


def make_scenarios(X, ratios):
    """For every row of X make one copy per candidate price ratio.
    Only price-related columns change; everything else is held fixed.
    Row order: all candidates of row 0, then all candidates of row 1, ..."""
    out = np.repeat(X, len(ratios), axis=0)
    r = np.tile(ratios, len(X))
    base = out[:, COL["base_price"]]
    discount = np.clip((1 - r) * 100, 0, None)
    nearest = np.abs(discount[:, None] - promo_levels[None, :]).argmin(axis=1)
    out[:, COL["price_ratio"]] = r
    out[:, COL["current_price"]] = base * r
    out[:, COL["discount_depth"]] = base - base * r
    out[:, COL["discount_pct"]] = discount
    out[:, COL["promotion_type_code"]] = promo_codes[nearest]
    return out


def scores(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "WAPE": float(np.abs(y - pred).sum() / y.sum()),
            "R2": r2_score(y, pred)}


X_train, y_train = train[FEATURES].to_numpy(float), train["units_sold"].to_numpy(float)
X_test, y_test = test[FEATURES].to_numpy(float), test["units_sold"].to_numpy(float)


results = {}

# (a) Naive: "tomorrow = yesterday"
results["Naive (yesterday's sales)"] = scores(y_test, test["lag1"])

# (b) Log-log regression: ln(units) = a + b * ln(price_ratio)   (b = elasticity)
pos = train[train["units_sold"] > 0]
x_ll = np.log(pos["price_ratio"]).to_numpy().reshape(-1, 1)
y_ll = np.log(pos["units_sold"]).to_numpy()
loglog = LinearRegression().fit(x_ll, y_ll)
smear = np.mean(np.exp(y_ll - loglog.predict(x_ll)))       # fixes the bias of exp(log-prediction)
loglog_pred = smear * np.exp(loglog.predict(np.log(test[["price_ratio"]].to_numpy())))
results["Log-log (price only)"] = scores(y_test, loglog_pred)

# (c) Gradient boosting. Choose the number of trees on the last 15% of the TRAIN dates
#     (never on the test period).
train_dates = np.sort(train["date"].unique())
val_cut = train_dates[int(len(train_dates) * 0.85)]
fit_part, val_part = train[train["date"] < val_cut], train[train["date"] >= val_cut]


def make_gbm(n_trees):
    # Same settings as before, written in XGBoost's names:
    # max_leaf_nodes=31 -> max_leaves=31 (leaf-wise growth), min_samples_leaf=50 -> min_child_weight=50,
    # l2_regularization -> reg_lambda, categorical columns flagged with feature_types ("c") + enable_categorical.
    return XGBRegressor(
        n_estimators=n_trees, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
        max_depth=0, max_leaves=31, min_child_weight=50, reg_lambda=1.0,
        enable_categorical=True, feature_types=["c" if c else "q" for c in CAT_MASK],
        random_state=RANDOM_STATE)


best_trees, best_rmse = 100, np.inf
for n_trees in (100, 200, 400, 600):
    m = make_gbm(n_trees).fit(fit_part[FEATURES].to_numpy(float), fit_part["units_sold"])
    rmse = scores(val_part["units_sold"], m.predict(val_part[FEATURES].to_numpy(float)))["RMSE"]
    print(f"[tune] trees={n_trees} validation RMSE={rmse:.4f}")
    if rmse < best_rmse:
        best_trees, best_rmse = n_trees, rmse

gbm = make_gbm(best_trees).fit(X_train, y_train)
results[f"Gradient boosting ({best_trees} trees)"] = scores(y_test, np.clip(gbm.predict(X_test), 0, None))

comparison = pd.DataFrame(results).T[["MAE", "RMSE", "WAPE", "R2"]]
print("\n=== Test-period comparison ===")
print(comparison.round(4))


def within_elasticity(data):
    d = data[data["units_sold"] > 0]
    x = np.log(d["price_ratio"])
    y = np.log(d["units_sold"])
    x = x - x.groupby(d["series"]).transform("mean")
    y = y - y.groupby(d["series"]).transform("mean")
    beta = float((x * y).sum() / (x * x).sum())
    resid = y - beta * x
    dof = len(d) - d["series"].nunique() - 1
    se = float(np.sqrt((resid ** 2).sum() / dof / (x * x).sum()))
    return beta, se


elasticity, elasticity_se = within_elasticity(df)
elasticity_by_category = {c: within_elasticity(g)[0] for c, g in df.groupby("category")}
print(f"\nElasticity (within series): {elasticity:.3f} +/- {elasticity_se:.3f}")
print("Pooled log-log slope (price only):", round(float(loglog.coef_[0]), 3))

# Sanity check: when price goes UP, does the model ever predict demand going UP?
rng = np.random.default_rng(RANDOM_STATE)
sample = X_test[rng.choice(len(X_test), 5000, replace=False)]
d_low = gbm.predict(make_scenarios(sample, np.array([0.85])))
d_high = gbm.predict(make_scenarios(sample, np.array([0.95])))
share_rises = float((d_high > d_low + 1e-9).mean())
print(f"Share of rows where predicted demand RISES when price goes 85% -> 95% of base: {share_rises:.1%}")


def optimise(X, lo, hi, n_prices=50, chunk=2000):
    ratios = np.linspace(lo, hi, n_prices)
    best_ratio = np.empty(len(X))
    best_revenue = np.empty(len(X))
    for s in range(0, len(X), chunk):
        sub = X[s:s + chunk]
        demand = np.clip(gbm.predict(make_scenarios(sub, ratios)), 0, None).reshape(len(sub), n_prices)
        price = sub[:, COL["base_price"]][:, None] * ratios[None, :]
        sold = np.minimum(demand, sub[:, COL["inventory_level"]][:, None])
        revenue = price * sold
        j = revenue.argmax(axis=1)
        best_ratio[s:s + chunk] = ratios[j]
        best_revenue[s:s + chunk] = revenue[np.arange(len(sub)), j]
    return best_ratio, best_revenue


actual_revenue = float(test["revenue"].sum())
pred_hist = np.clip(gbm.predict(X_test), 0, None)
model_revenue_now = float((test["current_price"].to_numpy()
                           * np.minimum(pred_hist, test["inventory_level"].to_numpy())).sum())
print(f"\nTest period revenue: actual {actual_revenue:,.0f} | model at historical prices {model_revenue_now:,.0f}")

lift = {}
for label, (lo, hi) in {"price_range_0.7_to_1.0": (0.7, 1.0)}.items():
    ratio, revenue = optimise(X_test, lo, hi)
    lift[label] = {
        "range": [lo, hi],
        "lift_vs_model_%": float(100 * (revenue.sum() / model_revenue_now - 1)),
        "lift_vs_actual_%": float(100 * (revenue.sum() / actual_revenue - 1)),
        "share_at_upper_bound_%": 100 * float((ratio >= hi - 1e-9).mean()),
        "mean_best_ratio": float(ratio.mean()),
    }
    print(label, {k: (round(v, 2) if not isinstance(v, list) else v) for k, v in lift[label].items()})


gbm = make_gbm(best_trees).fit(df[FEATURES].to_numpy(float), df["units_sold"])

catalog = {}                      # "product|region" -> latest known situation
for key, g in df.groupby("series"):
    last = g.iloc[-1]
    catalog[key] = {
        "product_id": last["product_id"], "region": last["region"],
        "category": last["category"], "brand": last["brand"],
        "base_price": float(last["base_price"]),
        "inventory_level": int(last["inventory_level"]),
        "roll7": float(g["units_sold"].tail(7).mean()),
        "roll14": float(g["units_sold"].tail(14).mean()),
    }

season_of_month = df.groupby("month")["season"].agg(lambda s: s.mode().iat[0]).to_dict()

bundle = {
    "model": gbm,
    "features": FEATURES,
    "maps": maps,
    "promo_levels": promo_levels,
    "promo_codes": promo_codes,
    "catalog": catalog,
    "season_of_month": season_of_month,
    "channels": sorted(df["channel"].unique().tolist()),
    "elasticity": elasticity,
    "elasticity_se": elasticity_se,
    "elasticity_by_category": elasticity_by_category,
    "metrics": {
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "n_trees": best_trees,
        "comparison": comparison.reset_index().rename(columns={"index": "Model"}).to_dict("records"),
        "share_demand_rises_when_price_rises": share_rises,
        "lift": lift,
        "actual_revenue": actual_revenue,
        "model_revenue_now": model_revenue_now,
    },
}
with open("pricing_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\nSaved pricing_model.pkl")