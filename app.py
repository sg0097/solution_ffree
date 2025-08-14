import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, prepare_category_group, compute_growth_rates, kpi_delta, filter_controls, trend_charts

st.set_page_config(page_title="India Vehicle Registrations – Investor Dashboard", page_icon="🚗", layout="wide")
st.title("🚗 India Vehicle Registrations – Investor Dashboard")
st.caption("Source: VAHAN (Parivahan) public data via India Data Portal • Built with Streamlit")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    show_ev_only = st.checkbox("EV only (where identifiable)", value=False)
    st.markdown("---")
    st.caption("Tip: Use the multiselects to slice the data by investor-relevant cohorts.")

# Load yearly data for YoY and maker-level analysis
df_yearly = load_data(ev_only=show_ev_only, monthly=False)
has_maker = bool(df_yearly.attrs.get("has_maker", False))

# Load monthly data for QoQ and category trends
df_monthly = load_data(ev_only=show_ev_only, monthly=True)

# Year range slider based on both datasets' years
min_year = min(df_yearly['date'].dt.year.min(), df_monthly['date'].dt.year.min())
max_year = max(df_yearly['date'].dt.year.max(), df_monthly['date'].dt.year.max())

with st.sidebar:
    start_year, end_year = st.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

# Filter datasets by year range
start_date = pd.to_datetime(f"{start_year}-01-01")
end_date = pd.to_datetime(f"{end_year}-12-31")
df_yearly = df_yearly[(df_yearly["date"] >= start_date) & (df_yearly["date"] <= end_date)].copy()
df_monthly = df_monthly[(df_monthly["date"] >= start_date) & (df_monthly["date"] <= end_date)].copy()

# Prepare vehicle group categories
mapper = prepare_category_group()
df_yearly["vehicle_group"] = df_yearly["category"].map(mapper).fillna("Other")
df_monthly["vehicle_group"] = df_monthly["category"].map(mapper).fillna("Other")

# Category options from monthly dataset (more complete)
cat_options = sorted(df_monthly["vehicle_group"].dropna().unique().tolist())

# Manufacturer options from yearly dataset (maker data only)
maker_options = sorted(df_yearly["maker"].dropna().unique().tolist()) if has_maker else []

# Sidebar multi-select filters
selected_cats, selected_makers = filter_controls(cat_options, maker_options, has_maker)

if selected_cats:
    df_yearly = df_yearly[df_yearly["vehicle_group"].isin(selected_cats)]
    df_monthly = df_monthly[df_monthly["vehicle_group"].isin(selected_cats)]
if has_maker and selected_makers:
    df_yearly = df_yearly[df_yearly["maker"].isin(selected_makers)]

if df_yearly.empty and df_monthly.empty:
    st.warning("No data matches the current filters. Try expanding the date range or disabling the EV-only toggle.")
    st.stop()

# Aggregations
yearly_agg = (df_yearly
              .groupby([pd.Grouper(key="date", freq="YS"), "vehicle_group"] + (["maker"] if has_maker else []), as_index=False)
              ["registrations"].sum())

monthly_agg = (df_monthly
               .groupby([pd.Grouper(key="date", freq="MS"), "vehicle_group"], as_index=False)
               ["registrations"].sum())

topline_q = monthly_agg.groupby([pd.Grouper(key="date", freq="Q"), "vehicle_group"], as_index=False)["registrations"].sum()
topline_y = yearly_agg.groupby([pd.Grouper(key="date", freq="YS"), "vehicle_group"], as_index=False)["registrations"].sum()

# Display KPIs with QoQ from monthly and YoY from yearly data
st.subheader("Market KPIs – QoQ (monthly data) & YoY (yearly data) growth by vehicle category")
ncols = max(1, min(4, len(cat_options)))
kpi_cols = st.columns(ncols)
for i, vg in enumerate(cat_options[:ncols]):
    qdf = topline_q[topline_q["vehicle_group"] == vg].set_index("date").sort_index()
    ydf = topline_y[topline_y["vehicle_group"] == vg].set_index("date").sort_index()

    qg = compute_growth_rates(qdf["registrations"], period="Q")
    yg = compute_growth_rates(ydf["registrations"], period="Y")

    with kpi_cols[i]:
        st.metric(label=f"{vg} – QoQ",
                  value=f"{int(qdf['registrations'].iloc[-1]) if len(qdf) > 0 else 0:,}",
                  delta=kpi_delta(qg))
        st.metric(label=f"{vg} – YoY",
                  value=f"{int(ydf['registrations'].iloc[-1]) if len(ydf) > 0 else 0:,}",
                  delta=kpi_delta(yg))

st.subheader("Trends – Total registrations by category (Monthly data)")
trend_charts(monthly_agg.rename(columns={"date": "Date", "registrations": "Registrations"}),
             line_by="vehicle_group", date_key="Date", value_key="Registrations",
             title="Monthly registrations by vehicle category")

if has_maker:
    st.subheader("Manufacturer cohorts – YoY (Yearly data only)")
    man_yearly = yearly_agg.groupby([pd.Grouper(key="date", freq="YS"), "maker"], as_index=False)["registrations"].sum()
    cutoff = man_yearly["date"].max() - pd.offsets.YearBegin(1) if len(man_yearly) > 0 else None
    top_makers = []
    if cutoff is not None:
        last12 = man_yearly[man_yearly["date"] >= cutoff]
        top_makers = (last12.groupby("maker")["registrations"].sum().sort_values(ascending=False).head(15).index.tolist())
    man_view = man_yearly[man_yearly["maker"].isin(top_makers)] if top_makers else man_yearly

    trend_charts(man_view.rename(columns={"date": "Date", "registrations": "Registrations", "maker": "Maker"}),
                 line_by="Maker", date_key="Date", value_key="Registrations",
                 title="Top manufacturers – yearly registrations")

    st.subheader("Growth table – YoY by manufacturer (yearly data only)")
    my = man_yearly.groupby([pd.Grouper(key="date", freq="YS"), "maker"], as_index=False)["registrations"].sum()

    def table_with_growth(gdf, period_label):
        out = []
        for m, sub in gdf.groupby("maker"):
            s = sub.set_index("date").sort_index()["registrations"]
            growth = compute_growth_rates(s, period="Y")
            out.append({"Maker": m,
                        f"{period_label} change %": None if growth is None else round(growth * 100, 1),
                        "Latest period": int(s.iloc[-1]) if len(s) > 0 else None})
        return pd.DataFrame(out).sort_values(by=f"{period_label} change %", ascending=False)

    st.dataframe(table_with_growth(my, "YoY"), use_container_width=True)

else:
    st.warning("Manufacturer column not found in yearly dataset; manufacturer-level analysis is hidden. Category trends and KPIs are still available.")

st.caption("Notes: QoQ uses calendar quarters on monthly dataset; YoY uses calendar years on yearly dataset. Source headers can vary; loader normalizes names automatically.")
