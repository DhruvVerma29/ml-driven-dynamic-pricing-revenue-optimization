
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dynamic Pricing Optimiser", layout="wide")


@st.cache_resource
def load_bundle():
    with open("pricing_model.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
maps = bundle["maps"]
catalog = bundle["catalog"]
promo_levels, promo_codes = bundle["promo_levels"], bundle["promo_codes"]
COL = {f: i for i, f in enumerate(FEATURES)}


def make_scenarios(X, ratios):
    """Same function as in Train.py: one copy of each row per candidate price ratio.
    Only price-related columns change; everything else is held fixed."""
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


st.title("Dynamic Pricing & Inventory-Constrained Optimiser")
st.caption("For one product in one region, on one day: try prices, predict units sold, "
           "respect the inventory limit, and find the best price. Other conditions are held fixed.")

# ------------------------------------------------------------------ sidebar inputs
st.sidebar.header("Situation")
products = sorted({v["product_id"] for v in catalog.values()})
product = st.sidebar.selectbox("Product", products)
regions = sorted(v["region"] for v in catalog.values() if v["product_id"] == product)
region = st.sidebar.selectbox("Region", regions)
item = catalog[product + "|" + region]
st.sidebar.caption(f"{item['category']} | {item['brand']} | base price ${item['base_price']:.2f}")

channel = st.sidebar.selectbox("Channel", bundle["channels"])
month = st.sidebar.selectbox("Month", sorted(bundle["season_of_month"]))
season = bundle["season_of_month"][month]
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow = st.sidebar.selectbox("Day of week", list(range(7)), index=2, format_func=lambda i: day_names[i])

inventory = st.sidebar.number_input("Units in stock", min_value=0, value=int(item["inventory_level"]))
roll7 = st.sidebar.number_input("Average daily units, last 7 days", min_value=0.0,
                                value=round(item["roll7"], 1))
roll14 = st.sidebar.number_input("Average daily units, last 14 days", min_value=0.0,
                                 value=round(item["roll14"], 1))
unit_cost = st.sidebar.number_input("Unit cost ($)", min_value=0.0, value=0.0,
                                    help="Not in the data. Leave 0 to maximise revenue; "
                                         "enter a cost to maximise profit instead.")

st.sidebar.header("Prices")
lo_pct, hi_pct = st.sidebar.slider("Prices to search (% of base price)", 70, 100, (70, 100))
cur_pct = st.sidebar.slider("Current price (% of base price)", 50, 100, 100, step=5)
if hi_pct > 100:
    st.sidebar.warning("The data never has a price above the base price, so results above 100% "
                       "are extrapolation and will look too good.")

# ------------------------------------------------------------------ one row describing the situation
base = item["base_price"]
row = np.zeros((1, len(FEATURES)))
values = {
    "base_price": base, "current_price": base * cur_pct / 100, "discount_depth": base * (1 - cur_pct / 100),
    "price_ratio": cur_pct / 100, "discount_pct": 100 - cur_pct, "inventory_level": inventory,
    "day_of_week": dow, "month": month, "is_weekend": int(dow >= 5), "roll7": roll7, "roll14": roll14,
    "category_code": maps["category"][item["category"]], "brand_code": maps["brand"][item["brand"]],
    "region_code": maps["region"][region], "channel_code": maps["channel"][channel],
    "season_code": maps["season"][season], "promotion_type_code": 0,   # label is set by make_scenarios
}
for name, v in values.items():
    row[0, COL[name]] = v

# ------------------------------------------------------------------ simulate prices
ratios = np.linspace(lo_pct / 100, hi_pct / 100, 101)
demand = np.clip(model.predict(make_scenarios(row, ratios)), 0, None)       # units people want
price = base * ratios
sold = np.minimum(demand, inventory)                                         # cannot sell more than stock
revenue = price * sold
profit = (price - unit_cost) * sold
goal_name = "Profit" if unit_cost > 0 else "Revenue"
goal = profit if unit_cost > 0 else revenue

best = int(np.argmax(goal))
now_ratio = cur_pct / 100
demand_now = float(np.clip(model.predict(make_scenarios(row, np.array([now_ratio]))), 0, None)[0])
sold_now = min(demand_now, inventory)
price_now = base * now_ratio
goal_now = (price_now - unit_cost) * sold_now if unit_cost > 0 else price_now * sold_now

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best price", f"${price[best]:.2f}", f"{100 * ratios[best]:.0f}% of base")
c2.metric("Units sold at best price", f"{sold[best]:.1f}", f"{sold[best] - sold_now:+.1f} vs now")
c3.metric(f"{goal_name} / day at best price", f"${goal[best]:.2f}", f"{goal[best] - goal_now:+.2f} vs now")
elasticity_cat = bundle["elasticity_by_category"][item["category"]]
c4.metric("Price elasticity (category)", f"{elasticity_cat:.2f}",
          help=f"All products: {bundle['elasticity']:.2f}. A 1% price rise changes units by about this many %.")

if hi_pct > 100:
    st.warning("The search range goes above the base price, which the model never saw. Treat this as a guess.")
if best in (0, len(ratios) - 1):
    st.info("The best price sits at the edge of the search range. Widening the range or testing it "
            "with a real experiment is the safe way to learn more.")
if demand[best] > inventory:
    st.info(f"Inventory is limiting sales here: people would buy {demand[best]:.1f} units but only "
            f"{inventory} are in stock.")

# ------------------------------------------------------------------ charts
left, right = st.columns(2)

fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=price, y=demand, name="Predicted demand", line=dict(color="orange")))
fig_d.add_hline(y=inventory, line_dash="dash", line_color="grey", annotation_text="stock")
fig_d.add_vline(x=price_now, line_dash="dot", annotation_text="current")
fig_d.update_layout(title="Demand vs price", xaxis_title="Price ($)", yaxis_title="Units per day", height=380)
left.plotly_chart(fig_d, use_container_width=True)

fig_r = go.Figure()
fig_r.add_trace(go.Scatter(x=price, y=goal, name=goal_name, line=dict(color="royalblue")))
fig_r.add_vline(x=price_now, line_dash="dot", annotation_text="current")
fig_r.add_vline(x=price[best], line_dash="dash", line_color="green", annotation_text="best")
fig_r.update_layout(title=f"{goal_name} vs price", xaxis_title="Price ($)",
                    yaxis_title=f"{goal_name} per day ($)", height=380)
right.plotly_chart(fig_r, use_container_width=True)

table = pd.DataFrame({"Price ($)": price, "% of base": 100 * ratios, "Demand": demand,
                      "Units sold": sold, "Revenue ($)": revenue, "Profit ($)": profit})
with st.expander("Numbers behind the charts"):
    st.dataframe(table.round(2), height=300)

# ------------------------------------------------------------------ model validation
with st.expander("How good is the model?"):
    m = bundle["metrics"]
    st.write(f"Trained on {m['train_period'][0]} to {m['train_period'][1]}; "
             f"tested on unseen days {m['test_period'][0]} to {m['test_period'][1]} "
             f"(time-based split). Gradient boosting with {m['n_trees']} trees.")
    st.dataframe(pd.DataFrame(m["comparison"]).set_index("Model").round(3))
    st.write(f"Predicted demand rose when the price rose in "
             f"{m['share_demand_rises_when_price_rises']:.1%} of tested cases.")
    st.write(f"Price elasticity (each product-region compared with itself): "
             f"**{bundle['elasticity']:.2f}** (standard error {bundle['elasticity_se']:.3f}). "
             "Demand is inelastic: a price cut brings less extra volume than the revenue it gives away.")
    lift = pd.DataFrame(m["lift"]).T[["lift_vs_model_%", "lift_vs_actual_%", "share_at_upper_bound_%"]]
    st.write("Revenue gain from choosing the best price for every test row:")
    st.dataframe(lift.round(2))
    st.markdown("**Limits:** historical (observational) data, not an experiment; only 90 days; "
                "the data has no prices above the base price; no costs unless you enter one; "
                "one day at a time, with no effect of today's price on tomorrow.")