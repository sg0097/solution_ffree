import streamlit as st
import pandas as pd
import numpy as np
import re

# Paths to your datasets (adjust accordingly)
VAHAN_MAKER_CSV_YEARLY = r"data/ffinal.csv"
VAHAN_MONTHLY_CSV = r"data/month.csv"

CANON = {
    "date": "date",
    "state": "state",
    "state name": "state",
    "state_name": "state",
    "rto": "rto",
    "rto name": "rto",
    "rto_name": "rto",
    "office_name": "rto",
    "maker": "maker",
    "type": "maker",
    "make": "maker",
    "make_name": "maker",
    "manufacturer": "maker",
    "company": "maker",
    "oem": "maker",
    "category": "category",
    "veh_category": "category",
    "vehicle_category": "category",
    "registrations": "registrations",
    "count": "registrations",
    "no_of_vehicles": "registrations",
    "total_vehicles": "registrations"
}

def _canonicalize_columns(cols):
    out = []
    for c in cols:
        c0 = str(c).strip().lower()
        c0 = re.sub(r"[\s\-]+", " ", c0)
        c0 = c0.replace("(nos.)","").strip()
        out.append(CANON.get(c0, c0))
    return out

@st.cache_data(show_spinner=True, ttl=60*60)
def load_data(ev_only: bool=False, monthly: bool=False) -> pd.DataFrame:
    if monthly:
        # Load monthly dataset
        df_wide = pd.read_csv(VAHAN_MONTHLY_CSV, low_memory=False)
        df_wide["date"] = pd.to_datetime(df_wide["Year"].astype(str) + "-" + df_wide["Month"], format="%Y-%b")
        df_wide = df_wide.drop(columns=["Year", "Month"])

        # Melt wide format to long form with 'category' and 'registrations'
        df = df_wide.melt(id_vars=["date"], var_name="category", value_name="registrations")

        # Ensure numeric types and fill NaNs
        df["registrations"] = pd.to_numeric(df["registrations"], errors="coerce").fillna(0)

        # No maker info in monthly data
        df["maker"] = np.nan
        df.attrs["has_maker"] = False

    else:
        # Load yearly dataset
        df = pd.read_csv(VAHAN_MAKER_CSV_YEARLY, low_memory=False)
        orig_cols = df.columns.tolist()
        df.columns = _canonicalize_columns(df.columns)
        df = df.loc[:, ~df.columns.duplicated()]
        required_min = {"date", "category", "registrations"}
        if not required_min.issubset(set(df.columns)):
            raise ValueError(f"CSV is missing required basics {required_min}. Found: {orig_cols} -> canonical: {df.columns.tolist()}")

        has_maker = "maker" in df.columns
        df["date"] = pd.to_datetime(df["date"], format='%Y', errors="coerce")
        df["registrations"] = pd.to_numeric(df["registrations"], errors="coerce").fillna(0)
        df = df.dropna(subset=["date", "category", "registrations"])
        for c in ["state","rto","maker","category"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        if ev_only:
            df = df[df["category"].str.contains("ELECTRIC|EV", case=False, na=False)]
        df.attrs["has_maker"] = has_maker

    return df

def prepare_category_group():
    import re
    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def mapper(raw):
        if not isinstance(raw, str) or not raw.strip():
            return "Other"
        r = norm(raw)
        # 2W
        if any(k in r for k in [
            "two wheeler", "twowheeler", "2w", "motor cycle", "motorcycle", "m cycle", "mcycle",
            "scooter", "sctr", "moped", "bike", "l1", "l2"
        ]):
            return "2W"
        # 3W
        if any(k in r for k in [
            "three wheeler", "threewheeler", "3w", "auto rickshaw", "autorickshaw", "rickshaw",
            "e rickshaw", "erickshaw", "l5", "e rick"
        ]):
            return "3W"
        # 4W
        if any(k in r for k in [
            "four wheeler", "fourwheeler", "4w", "lmv", "car", "motor car", "passenger car",
            "jeep", "van", "suv", "quadricycle", "qute", "lgv", "lcv", "mcv", "hcv", "hgv",
            "goods", "goods carrier", "truck", "bus", "omni bus", "omnibus", "taxi", "cab",
            "pickup", "tractor", "tempo", "lorry"
        ]):
            return "4W"
        return "Other"
    return mapper

def compute_growth_rates(series: pd.Series, period: str="Q"):
    if series is None or len(series) < 2:
        return None
    s = series.sort_index().dropna()
    if len(s) < 2:
        return None
    prev, curr = s.iloc[-2], s.iloc[-1]
    if prev == 0:
        return None
    return (curr - prev) / prev

def kpi_delta(growth):
    if growth is None or np.isinf(growth) or np.isnan(growth):
        return "n/a"
    sign = "+" if growth >= 0 else ""
    return f"{sign}{round(growth*100,1)}%"

def filter_controls(cat_options, maker_options, has_maker: bool):
    c1, c2 = st.sidebar.columns(2)
    with c1:
        selected_cats = st.multiselect("Vehicle category", cat_options, default=cat_options)
    selected_makers = []
    if has_maker:
        with c2:
            selected_makers = st.multiselect("Manufacturers", maker_options[:10], default=[])
    else:
        st.sidebar.info("Manufacturer field not present in this dataset; maker-level analysis is hidden.")
    return selected_cats, selected_makers

def trend_charts(df, line_by: str, date_key: str, value_key: str, title: str):
    import altair as alt
    base = alt.Chart(df).mark_line(point=False).encode(
        x=alt.X(f"{date_key}:T", title="Month"),
        y=alt.Y(f"{value_key}:Q", title="Registrations"),
        color=alt.Color(f"{line_by}:N", legend=alt.Legend(title=line_by.replace('_',' ').title()))
    ).properties(title=title, height=380)
    st.altair_chart(base, use_container_width=True)
