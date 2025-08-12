import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Enhanced UI imports for colourful dashboard
try:
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.badges import badge
    from streamlit_option_menu import option_menu
    from streamlit_card import card
    from streamlit_lottie import st_lottie
except Exception:  # pragma: no cover
    style_metric_cards = None
    colored_header = None
    badge = None
    option_menu = None
    card = None
    st_lottie = None

# Guard optional plotting libs to avoid hard failures
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

try:
    import streamlit as st
except Exception as e:  # pragma: no cover
    raise SystemExit("Streamlit must be installed to run this app: pip install streamlit plotly pandas numpy") from e

# Optional analytics dependencies
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except Exception:  # pragma: no cover
    seasonal_decompose = None

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover
    Prophet = None

try:
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
except Exception:  # pragma: no cover
    Ridge = None
    train_test_split = None


# -------------------------------
# Data loading and utilities
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_FILES = {
    "marketing": BASE_DIR / "doordash powers marketing.csv",
    "operations": BASE_DIR / "doordash powers operations.csv",
    "payouts": BASE_DIR / "doordash powers payouts.csv",
    "sales": BASE_DIR / "doordash powers sales.csv",
}


def _safe_lower(text: str) -> str:
    return text.lower().strip()


def find_column_by_keywords(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first dataframe column whose lowercase name contains any of the keywords (lowercased).
    Useful when exact column names may vary slightly across exports.
    """
    lowered = {col: _safe_lower(col) for col in df.columns}
    for col, low in lowered.items():
        for kw in keywords:
            if _safe_lower(kw) in low:
                return col
    return None


def safe_divide(numerator, denominator):
    """Robust element-wise divide that tolerates zeros/NaNs and scalar denominators.

    Rules:
    - If denominator is 0 or NaN → return numerator
    - Otherwise → return numerator / denominator
    Works for scalars, Series, and DataFrames without shape-mismatch errors.
    """
    # Normalize inputs to pandas objects when possible for consistent behavior
    num = numerator
    den = denominator
    if not isinstance(num, (pd.Series, pd.DataFrame)):
        num = pd.to_numeric(num, errors="coerce")
    if not isinstance(den, (pd.Series, pd.DataFrame)):
        den = pd.to_numeric(den, errors="coerce")

    # Fast-path: scalar denominator (common in min-max normalization)
    if np.isscalar(den) or (isinstance(den, (pd.Series, pd.DataFrame)) and den.size == 1):
        den_scalar = den if np.isscalar(den) else (den.values.item() if hasattr(den, "values") else float(den))
        if pd.isna(den_scalar) or den_scalar == 0:
            return num
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den_scalar

    # General case: align shapes where possible
    aligned_den = den
    if isinstance(num, pd.Series) and isinstance(den, pd.Series):
        aligned_den = den.reindex_like(num)
    elif isinstance(num, pd.DataFrame) and isinstance(den, pd.DataFrame):
        aligned_den = den.reindex_like(num)

    mask = (aligned_den == 0) | pd.isna(aligned_den)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / aligned_den

    # Where denominator invalid, fall back to numerator
    if isinstance(result, (pd.Series, pd.DataFrame)):
        result = result.mask(mask, num)
        result = result.replace([np.inf, -np.inf], np.nan).mask(mask, num)
    else:
        if bool(np.any(mask)):
            result = num
        if np.isinf(result):
            result = num
    return result


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # Fallback for potential encoding issues
        return pd.read_csv(path, encoding="latin-1")


def parse_datetime_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
    return df


def add_week_start(df: pd.DataFrame, date_col: str, new_col: str = "Week") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.copy()
    series = pd.to_datetime(df[date_col], errors="coerce")
    df[new_col] = series.dt.to_period("W-MON").apply(lambda r: r.start_time)
    return df


def filter_df_by_date(df: pd.DataFrame, date_col: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return df.loc[mask]


def make_metric_card(label: str, value, delta: Optional[str] = None, help_text: Optional[str] = None):
    col = st.container()
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


# -------------------------------
# Section: Marketing
# -------------------------------

def section_marketing(marketing_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="📢 Marketing Analytics",
            description="Campaign performance, customer acquisition, and ROI analysis",
            color_name="green-70"
        )
    else:
        st.markdown("## 📢 Marketing Analytics")
        st.markdown("*Campaign performance, customer acquisition, and ROI analysis*")
    
    if marketing_df.empty:
        st.info("📭 Marketing CSV not found or empty.")
        return

    # Parse and prepare
    marketing_df = parse_datetime_column(marketing_df, "Date")

    # Sidebar pre/post ranges (defaults match notebook)
    st.sidebar.markdown("**Marketing period comparison**")
    default_pre_start = pd.Timestamp("2025-06-01")
    default_pre_end = pd.Timestamp("2025-07-02")
    default_post_start = pd.Timestamp("2025-07-03")
    default_post_end = pd.Timestamp("2025-08-03")

    pre_range = st.sidebar.date_input(
        "Pre period",
        value=(default_pre_start, default_pre_end),
        key="mk_pre_range",
    )
    post_range = st.sidebar.date_input(
        "Post period",
        value=(default_post_start, default_post_end),
        key="mk_post_range",
    )

    # Metrics of interest
    metrics = [
        "Orders",
        "Sales",
        "Customer Discounts from Marketing | (Funded by you)",
        "Marketing Fees | (Including any applicable taxes)",
        "Average Order Value",
        "ROAS",
        "New Customers Acquired",
        "Total Customers Acquired",
    ]

    # Build daily view (many notebook charts use `daily`)
    daily = marketing_df.copy()
    if "Date" in daily.columns:
        daily = daily.sort_values("Date")
        # If multiple rows per date, aggregate by sum/mean mix where appropriate
        agg_map = {}
        for m in metrics:
            if m in daily.columns:
                # numeric metrics -> sum; averages -> mean
                if "Average" in m:
                    agg_map[m] = "mean"
                else:
                    agg_map[m] = "sum"
        if agg_map:
            daily = (
                daily.groupby("Date", as_index=False)
                [list(agg_map.keys())]
                .agg(agg_map)
            )
    else:
        st.warning("Marketing data missing 'Date' column; some charts may be unavailable.")

    # KPI cards
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        total_orders = daily.get("Orders", pd.Series(dtype=float)).sum()
        st.metric("Total Orders", f"{int(total_orders):,}")
    with kpi_cols[1]:
        total_sales = daily.get("Sales", pd.Series(dtype=float)).sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with kpi_cols[2]:
        aov = daily.get("Average Order Value", pd.Series(dtype=float)).mean()
        if pd.notnull(aov):
            st.metric("Average Order Value", f"${aov:,.2f}")
        else:
            st.metric("Average Order Value", "—")
    with kpi_cols[3]:
        new_customers = daily.get("New Customers Acquired", pd.Series(dtype=float)).sum()
        st.metric("New Customers", f"{int(new_customers):,}")

    # Pre/Post comparison table
    if "Date" in marketing_df.columns:
        pre_start, pre_end = pre_range if isinstance(pre_range, tuple) else (None, None)
        post_start, post_end = post_range if isinstance(post_range, tuple) else (None, None)

        pre_df = filter_df_by_date(marketing_df, "Date", pd.to_datetime(pre_start), pd.to_datetime(pre_end))
        post_df = filter_df_by_date(marketing_df, "Date", pd.to_datetime(post_start), pd.to_datetime(post_end))

        def agg_series(df: pd.DataFrame) -> pd.Series:
            vals = {}
            for m in metrics:
                if m not in df.columns:
                    continue
                if "Average" in m:
                    vals[m] = df[m].mean()
                else:
                    vals[m] = df[m].sum()
            return pd.Series(vals)

        pre_summary = agg_series(pre_df).rename("Pre")
        post_summary = agg_series(post_df).rename("Post")
        comparison = pd.concat([pre_summary, post_summary], axis=1)
        comparison["Δ Absolute"] = comparison["Post"] - comparison["Pre"]
        comparison["Δ % Change"] = safe_divide(comparison["Δ Absolute"], comparison["Pre"]) * 100

        # Ensure numeric dtypes to avoid formatting bugs and then build a display copy
        numeric_cols = ["Pre", "Post", "Δ Absolute", "Δ % Change"]
        comparison_numeric = comparison.copy()
        for c in numeric_cols:
            comparison_numeric[c] = pd.to_numeric(comparison_numeric[c], errors="coerce")

        def fmt_int(x):
            return "—" if pd.isna(x) else f"{x:,.0f}"

        def fmt_pct(x):
            return "—" if pd.isna(x) else f"{x:.2f}%"

        display_df = comparison_numeric.copy()
        display_df["Pre"] = comparison_numeric["Pre"].map(fmt_int)
        display_df["Post"] = comparison_numeric["Post"].map(fmt_int)
        display_df["Δ Absolute"] = comparison_numeric["Δ Absolute"].map(fmt_int)
        display_df["Δ % Change"] = comparison_numeric["Δ % Change"].map(fmt_pct)

        st.markdown("**Pre vs Post comparison**")
        st.dataframe(display_df, use_container_width=True)

    # Time-series charts
    if px is not None and "Date" in daily.columns:
        ts_cols = [c for c in ["Orders", "Sales", "Average Order Value", "New Customers Acquired"] if c in daily.columns]
        if ts_cols:
            st.markdown("**Daily trends**")
            for col in ts_cols:
                fig = px.line(daily, x="Date", y=col, title=col)
                st.plotly_chart(fig, use_container_width=True)

        # Normalized multi-series
        norm_metrics = [c for c in ["Orders", "Sales", "Average Order Value", "ROAS", "New Customers Acquired"] if c in daily.columns]
        if len(norm_metrics) >= 2:
            base = daily[["Date"] + norm_metrics].set_index("Date").astype(float)
            normalized = pd.DataFrame(index=base.index)
            for c in norm_metrics:
                col_min = base[c].min()
                col_range = base[c].max() - col_min
                normalized[c] = safe_divide(base[c] - col_min, col_range)
            norm_long = normalized.reset_index().melt(id_vars="Date", var_name="Metric", value_name="Normalized")
            fig = px.line(norm_long, x="Date", y="Normalized", color="Metric", title="Daily Metrics (Min–Max Normalized)")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        corr_metrics = [c for c in ["Orders", "Sales", "Average Order Value", "ROAS", "New Customers Acquired"] if c in daily.columns]
        if len(corr_metrics) >= 2 and go is not None:
            corr = daily[corr_metrics].corr()
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1))
            heat.update_layout(title="Metric Correlation")
            st.plotly_chart(heat, use_container_width=True)

    # Seasonal decomposition (Orders)
    if seasonal_decompose is not None and "Orders" in daily.columns and "Date" in daily.columns:
        st.markdown("**Orders: weekly seasonal decomposition**")
        ts = daily.set_index("Date")["Orders"].asfreq("D")
        try:
            decomp = seasonal_decompose(ts, model="additive", period=7)
            st.line_chart(pd.DataFrame({
                "Observed": decomp.observed,
                "Trend": decomp.trend,
                "Seasonal": decomp.seasonal,
                "Resid": decomp.resid,
            }))
        except Exception:
            st.warning("Unable to compute seasonal decomposition.")
    else:
        if seasonal_decompose is None:
            st.info("Install statsmodels to see seasonal decomposition: pip install statsmodels")

    # Prophet forecast (Orders)
    if Prophet is not None and "Orders" in daily.columns and "Date" in daily.columns:
        st.markdown("**Orders forecast (Prophet)**")
        try:
            df_prophet = daily.rename(columns={"Date": "ds", "Orders": "y"})[["ds", "y"]]
            m = Prophet(daily_seasonality=True)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=30)
            fc = m.predict(future)
            # Lightweight display using Plotly if available
            if px is not None:
                fig = px.line(fc, x="ds", y="yhat", title="30-day Forecast of Orders")
                fig.add_scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Actual", mode="lines")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(fc.set_index("ds")["yhat"])  # fallback
        except Exception:
            st.warning("Prophet forecast failed to compute.")
    else:
        if Prophet is None:
            st.info("Install prophet to enable forecasting: pip install prophet")

    # Simple Ridge regression Sales ~ (Marketing Credit, Marketing Fees)
    if Ridge is not None and train_test_split is not None and not daily.empty:
        # Try to locate columns by fuzzy keywords
        mk_credit = find_column_by_keywords(marketing_df, ["marketing credit"]) or ""
        mk_fees = find_column_by_keywords(marketing_df, ["marketing fees"]) or ""
        target = "Sales" if "Sales" in marketing_df.columns else None
        candidate_cols = [c for c in [mk_credit, mk_fees] if c]
        if target and len(candidate_cols) >= 2:
            st.markdown("**Ridge regression: Sales vs Marketing spend**")
            try:
                # Aggregate to daily to avoid duplicate indices
                model_df = marketing_df[["Date", target] + candidate_cols].dropna()
                X = model_df[candidate_cols]
                y = model_df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
                model = Ridge(alpha=1.0).fit(X_train, y_train)
                r2 = model.score(X_test, y_test)
                coef_table = pd.DataFrame({"Feature": candidate_cols, "Coefficient": model.coef_})
                st.write(f"R²: {r2:.3f}")
                st.dataframe(coef_table)
            except Exception:
                st.warning("Could not fit Ridge regression model on available marketing columns.")
    else:
        if Ridge is None:
            st.info("Install scikit-learn to enable regression: pip install scikit-learn")


# -------------------------------
# Section: Operations
# -------------------------------

def section_operations(ops_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="⚙️ Operations Analytics",
            description="Store performance, ratings, and operational efficiency metrics",
            color_name="purple-70"
        )
    else:
        st.markdown("## ⚙️ Operations Analytics")
        st.markdown("*Store performance, ratings, and operational efficiency metrics*")
    
    if ops_df.empty:
        st.info("📭 Operations CSV not found or empty.")
        return

    ops_df = parse_datetime_column(ops_df, "Start Date")
    ops_df = parse_datetime_column(ops_df, "End Date")
    ops_df = add_week_start(ops_df, "Start Date", new_col="Week")

    # KPIs and derived metrics
    kpis = [
        "Total Orders Including Cancelled Orders",
        "Total Delivered or Picked Up Orders",
        "Total Missing or Incorrect Orders",
        "Total Error Charges",
        "Total Cancelled Orders",
        "Total Downtime in Minutes",
        "Average Rating",
    ]

    # Derived
    ops_df = ops_df.copy()
    if (
        "Total Cancelled Orders" in ops_df.columns and
        "Total Orders Including Cancelled Orders" in ops_df.columns
    ):
        ops_df["Cancellation Rate"] = safe_divide(
            ops_df["Total Cancelled Orders"], ops_df["Total Orders Including Cancelled Orders"]
        )
    if (
        "Total Downtime in Minutes" in ops_df.columns and
        "Total Orders Including Cancelled Orders" in ops_df.columns
    ):
        ops_df["Downtime per Order (min)"] = safe_divide(
            ops_df["Total Downtime in Minutes"], ops_df["Total Orders Including Cancelled Orders"]
        )

    # KPI cards
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        delivered = ops_df.get("Total Delivered or Picked Up Orders", pd.Series(dtype=float)).sum()
        st.metric("Delivered Orders", f"{int(delivered):,}")
    with kpi_cols[1]:
        cancel_rate = ops_df.get("Cancellation Rate", pd.Series(dtype=float)).mean()
        if pd.notnull(cancel_rate):
            st.metric("Avg Cancellation Rate", f"{cancel_rate*100:.2f}%")
        else:
            st.metric("Avg Cancellation Rate", "—")
    with kpi_cols[2]:
        downtime = ops_df.get("Downtime per Order (min)", pd.Series(dtype=float)).mean()
        if pd.notnull(downtime):
            st.metric("Avg Downtime / Order", f"{downtime:.2f} min")
        else:
            st.metric("Avg Downtime / Order", "—")
    with kpi_cols[3]:
        avg_rating = ops_df.get("Average Rating", pd.Series(dtype=float)).mean()
        if pd.notnull(avg_rating):
            st.metric("Average Rating", f"{avg_rating:.2f}")
        else:
            st.metric("Average Rating", "—")

    # Store summary (sum & mean)
    if {"Store ID", "Store Name"}.issubset(ops_df.columns):
        numeric_cols = [c for c in kpis if c in ops_df.columns]
        if numeric_cols:
            store_summary = (
                ops_df.groupby(["Store ID", "Store Name"])[numeric_cols]
                .agg(["sum", "mean"])
            )
            # Flatten columns
            store_summary.columns = ["_".join(col).strip() for col in store_summary.columns]
            st.markdown("**Store summary (sum/mean)**")
            st.dataframe(store_summary.reset_index())

    # Weekly WoW changes using safe division
    if "Week" in ops_df.columns and {"Store ID", "Store Name"}.issubset(ops_df.columns):
        wow_metrics = [
            c for c in [
                "Total Delivered or Picked Up Orders",
                "Total Cancelled Orders",
                "Total Missing or Incorrect Orders",
                "Total Downtime in Minutes",
            ] if c in ops_df.columns
        ]
        if wow_metrics:
            weekly = (
                ops_df.groupby(["Store ID", "Store Name", "Week"])[wow_metrics]
                .sum()
                .reset_index()
                .sort_values(["Store ID", "Week"])  # ensure chronological per store
            )
            for m in wow_metrics:
                prev = weekly.groupby("Store ID")[m].shift(1)
                numer = weekly[m] - prev
                weekly[f"{m} WoW %"] = safe_divide(numer, prev) * 100

            st.markdown("**WoW % change by store (Delivered Orders)**")
            target_col = "Total Delivered or Picked Up Orders WoW %"
            if px is not None and target_col in weekly.columns:
                wow_pivot = weekly.pivot(index="Week", columns="Store ID", values=target_col)
                fig = px.line(wow_pivot, x=wow_pivot.index, y=wow_pivot.columns, title="WoW % Change in Delivered Orders by Store")
                st.plotly_chart(fig, use_container_width=True)

            # Top 5 gainers/losers by last available week
            if target_col in weekly.columns:
                last_week = weekly["Week"].max()
                latest = weekly[weekly["Week"] == last_week]
                latest = latest[["Store Name", target_col]].dropna()
                gainers = latest.nlargest(5, target_col)
                losers = latest.nsmallest(5, target_col)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top 5 WoW gainers**")
                    st.dataframe(gainers.set_index("Store Name"))
                with c2:
                    st.markdown("**Top 5 WoW losers**")
                    st.dataframe(losers.set_index("Store Name"))

    # Time series of Delivered orders by store
    if px is not None and {"Week", "Store ID", "Store Name", "Total Delivered or Picked Up Orders"}.issubset(ops_df.columns):
        ts = (
            ops_df.groupby(["Store Name", "Week"])  
            ["Total Delivered or Picked Up Orders"].sum().reset_index()
        )
        st.markdown("**Delivered orders over time (by store)**")
        fig = px.line(ts, x="Week", y="Total Delivered or Picked Up Orders", color="Store Name")
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Section: Sales
# -------------------------------

def section_sales(sales_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="💰 Sales Analytics",
            description="Revenue analysis, order trends, and financial performance",
            color_name="orange-70"
        )
    else:
        st.markdown("## 💰 Sales Analytics")
        st.markdown("*Revenue analysis, order trends, and financial performance*")
    
    if sales_df.empty:
        st.info("📭 Sales CSV not found or empty.")
        return

    sales_df = parse_datetime_column(sales_df, "Start Date")
    sales_df = parse_datetime_column(sales_df, "End Date")
    sales_df = add_week_start(sales_df, "Start Date", new_col="Week")

    # Derived metrics mirroring the notebook
    def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
        return safe_divide(numer, denom)

    if {
        "Total Orders Including Cancelled Orders",
        "Total Delivered or Picked Up Orders",
        "Gross Sales",
    }.issubset(sales_df.columns):
        sales_df = sales_df.assign(
            Cancellation_Rate=safe_ratio(
                sales_df["Total Orders Including Cancelled Orders"] - sales_df["Total Delivered or Picked Up Orders"],
                sales_df["Total Orders Including Cancelled Orders"],
            ),
            Fulfillment_Rate=safe_ratio(
                sales_df["Total Delivered or Picked Up Orders"],
                sales_df["Total Orders Including Cancelled Orders"],
            ),
            Commission_Rate=safe_ratio(sales_df.get("Total Commission"), sales_df.get("Gross Sales")),
            Promo_ROI=safe_ratio(
                sales_df.get("Total Promotion Sales | (for historical reference only)"),
                sales_df.get("Total Promotion Fees | (for historical reference only)"),
            ),
            Ad_ROI=safe_ratio(
                sales_df.get("Total Ad Sales | (for historical reference only)"),
                sales_df.get("Total Ad Fees | (for historical reference only)"),
            ),
            Revenue_per_Delivered=safe_ratio(
                sales_df.get("Gross Sales"), sales_df.get("Total Delivered or Picked Up Orders")
            ),
        )

    # KPIs
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        gross = sales_df.get("Gross Sales", pd.Series(dtype=float)).sum()
        st.metric("Gross Sales", f"${gross:,.0f}")
    with kpi_cols[1]:
        delivered = sales_df.get("Total Delivered or Picked Up Orders", pd.Series(dtype=float)).sum()
        st.metric("Delivered Orders", f"{int(delivered):,}")
    with kpi_cols[2]:
        aov = sales_df.get("AOV", pd.Series(dtype=float)).mean()
        st.metric("AOV", f"${aov:,.2f}" if pd.notnull(aov) else "—")
    with kpi_cols[3]:
        fulfill_rate = sales_df.get("Fulfillment_Rate", pd.Series(dtype=float)).mean()
        st.metric("Avg Fulfillment Rate", f"{fulfill_rate*100:.2f}%" if pd.notnull(fulfill_rate) else "—")

    # Weekly aggregation
    if {"Store Name", "Week"}.issubset(sales_df.columns):
        weekly = sales_df.groupby(["Store Name", "Week"]).agg(
            {
                k: "sum"
                for k in [
                    "Gross Sales",
                    "Total Orders Including Cancelled Orders",
                    "Total Delivered or Picked Up Orders",
                ]
                if k in sales_df.columns
            }
        )
        if "AOV" in sales_df.columns:
            weekly["AOV"] = (
                sales_df.groupby(["Store Name", "Week"])  
                ["AOV"].mean()
            )
        for k in [
            "Cancellation_Rate",
            "Fulfillment_Rate",
            "Commission_Rate",
            "Promo_ROI",
            "Ad_ROI",
            "Revenue_per_Delivered",
        ]:
            if k in sales_df.columns:
                weekly[k] = sales_df.groupby(["Store Name", "Week"])[k].mean()
        weekly = weekly.reset_index()

        if px is not None and "Gross Sales" in weekly.columns:
            st.markdown("**Weekly Gross Sales per Store**")
            fig = px.line(weekly, x="Week", y="Gross Sales", color="Store Name")
            st.plotly_chart(fig, use_container_width=True)

        # Top 5 stores by total Gross Sales
        if px is not None and "Gross Sales" in weekly.columns:
            totals = weekly.groupby("Store Name")["Gross Sales"].sum().nlargest(5)
            top5 = totals.index.tolist()
            st.markdown("**Top 5 Stores: Weekly Gross Sales**")
            fig = px.line(weekly[weekly["Store Name"].isin(top5)], x="Week", y="Gross Sales", color="Store Name")
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: Promo Fees vs Promo Sales
        fees_col = "Total Promotion Fees | (for historical reference only)"
        sales_col = "Total Promotion Sales | (for historical reference only)"
        if px is not None and fees_col in sales_df.columns and sales_col in sales_df.columns:
            st.markdown("**Promo Spend vs Promo Sales**")
            try:
                fig = px.scatter(sales_df, x=fees_col, y=sales_col, trendline="ols")
            except Exception:
                fig = px.scatter(sales_df, x=fees_col, y=sales_col)
            st.plotly_chart(fig, use_container_width=True)

        # Bar: Avg Fulfillment Rate by Store
        if px is not None and "Fulfillment_Rate" in weekly.columns:
            st.markdown("**Store Fulfillment Rate Ranking**")
            fulfill_avg = weekly.groupby("Store Name")["Fulfillment_Rate"].mean().sort_values(ascending=False)
            fig = px.bar(fulfill_avg, orientation="v", labels={"value": "Avg Fulfillment Rate", "index": "Store Name"})
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Section: Payouts
# -------------------------------

def section_payouts(payout_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="💸 Payouts Analytics",
            description="Payment analysis, commission tracking, and financial settlements",
            color_name="red-70"
        )
    else:
        st.markdown("## 💸 Payouts Analytics")
        st.markdown("*Payment analysis, commission tracking, and financial settlements*")
    
    if payout_df.empty:
        st.info("📭 Payouts CSV not found or empty.")
        return

    payout_df = parse_datetime_column(payout_df, "Payout Date")

    # KPI cards
    kpi_cols = st.columns(4)
    net_payout_col = find_column_by_keywords(payout_df, ["net payout"]) or "Net Payout"
    subtotal_col = find_column_by_keywords(payout_df, ["subtotal"]) or "Subtotal"
    commission_col = find_column_by_keywords(payout_df, ["commission"]) or "Commission"
    mk_fee_col = find_column_by_keywords(payout_df, ["marketing fees"]) or "Marketing Fees | (Including any applicable taxes)"

    with kpi_cols[0]:
        net_payout = payout_df.get(net_payout_col, pd.Series(dtype=float)).sum()
        st.metric("Net Payout", f"${net_payout:,.0f}")
    with kpi_cols[1]:
        subtotal = payout_df.get(subtotal_col, pd.Series(dtype=float)).sum()
        st.metric("Subtotal", f"${subtotal:,.0f}")
    with kpi_cols[2]:
        commission = payout_df.get(commission_col, pd.Series(dtype=float)).sum()
        st.metric("Commission", f"${commission:,.0f}")
    with kpi_cols[3]:
        mk_fees = payout_df.get(mk_fee_col, pd.Series(dtype=float)).sum()
        st.metric("Marketing Fees", f"${mk_fees:,.0f}")

    # Group by Store and Date
    if {"Store Name", "Payout Date"}.issubset(payout_df.columns):
        metrics = [
            c for c in [
                net_payout_col,
                subtotal_col,
                commission_col,
                "Drive Charge",
                mk_fee_col,
                "Customer Discounts from Marketing | (Funded by You)",
            ] if c in payout_df.columns
        ]
        grouped = payout_df.groupby(["Store Name", "Payout Date"])[metrics].sum().reset_index()

        stores = ["All Stores"] + sorted(grouped["Store Name"].unique().tolist())
        
        # Create store selection with buttons
        st.markdown("**Select Store:**")
        
        # Create a grid of buttons for store selection
        cols = st.columns(3)  # 3 columns for better layout
        
        selected_store = "All Stores"  # default
        
        for i, store in enumerate(stores):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(store, key=f"store_{i}", use_container_width=True):
                    selected_store = store
        
        # Apply the selection
        if selected_store != "All Stores":
            data = grouped[grouped["Store Name"] == selected_store].sort_values("Payout Date")
        else:
            data = grouped.groupby("Payout Date")[metrics].sum().reset_index().sort_values("Payout Date")

        if px is not None and not data.empty:
            st.markdown("**Payout components over time**")
            fig = px.line(data, x="Payout Date", y=metrics, title=(selected_store if selected_store != "All Stores" else "All Stores"))
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Overview cards combining key stats
# -------------------------------

def section_overview(marketing_df: pd.DataFrame, ops_df: pd.DataFrame, sales_df: pd.DataFrame, payout_df: pd.DataFrame):
    # Colourful section header
    if colored_header:
        colored_header(
            label="📊 Data Overview Dashboard",
            description="Comprehensive insights across all DoorDash datasets",
            color_name="blue-70"
        )
    else:
        st.markdown("## 📊 Data Overview Dashboard")
        st.markdown("*Comprehensive insights across all DoorDash datasets*")
    
    # Enhanced KPI cards with icons and colours
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Calculate metrics
    total_orders = np.nan
    if "Total Delivered or Picked Up Orders" in sales_df.columns:
        total_orders = pd.to_numeric(sales_df["Total Delivered or Picked Up Orders"], errors="coerce").sum()
    elif "Orders" in marketing_df.columns:
        total_orders = pd.to_numeric(marketing_df["Orders"], errors="coerce").sum()
    val = int(total_orders) if pd.notnull(total_orders) else 0
    
    gross_sales = pd.to_numeric(sales_df.get("Gross Sales", pd.Series(dtype=float)), errors="coerce").sum()
    avg_rating = pd.to_numeric(ops_df.get("Average Rating", pd.Series(dtype=float)), errors="coerce").mean()
    net_payout_col = find_column_by_keywords(payout_df, ["net payout"]) or "Net Payout"
    net_payout = pd.to_numeric(payout_df.get(net_payout_col, pd.Series(dtype=float)), errors="coerce").sum()
    
    # Use regular metrics with styling
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("📦 Total Orders", f"{val:,}")
    
    with c2:
        st.metric("💰 Gross Sales", f"${gross_sales:,.0f}")
    
    with c3:
        st.metric("⭐ Average Rating", f"{avg_rating:.2f}" if pd.notnull(avg_rating) else "—")
    
    with c4:
        st.metric("💸 Net Payout", f"${net_payout:,.0f}")
    
    # Apply metric card styling if available
    if style_metric_cards:
        style_metric_cards()

    # Data Overview with colourful cards
    st.markdown("### 📁 Data Overview")
    
    # Calculate data overview metrics
    files_count = sum([
        1 if not df.empty else 0 
        for df in [marketing_df, ops_df, sales_df, payout_df]
    ])
    
    total_rows = sum([
        len(df) for df in [marketing_df, ops_df, sales_df, payout_df]
    ])
    
    total_columns = sum([
        len(df.columns) for df in [marketing_df, ops_df, sales_df, payout_df]
    ])
    
    all_stores = set()
    for df in [marketing_df, ops_df, sales_df, payout_df]:
        if "Store Name" in df.columns:
            all_stores.update(df["Store Name"].dropna().unique())
        if "Store ID" in df.columns:
            all_stores.update(df["Store ID"].dropna().astype(str).unique())
    
    # Use colourful cards if available
    if card:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            card(
                title="📂 Files Loaded",
                text=f"{files_count}/4 datasets successfully loaded",
                image="https://img.icons8.com/color/96/000000/folder-invoices.png",
                url=None
            )
        
        with col2:
            card(
                title="📊 Total Rows",
                text=f"{total_rows:,} data points across all files",
                image="https://img.icons8.com/color/96/000000/data-configuration.png",
                url=None
            )
        
        with col3:
            card(
                title="📋 Total Columns",
                text=f"{total_columns} unique data fields",
                image="https://img.icons8.com/color/96/000000/columns.png",
                url=None
            )
        
        with col4:
            card(
                title="🏪 Unique Stores",
                text=f"{len(all_stores)} distinct store locations",
                image="https://img.icons8.com/color/96/000000/shop.png",
                url=None
            )
    else:
        # Fallback to regular metrics
        c5, c6, c7, c8 = st.columns(4)
        
        with c5:
            st.metric("📂 Files Loaded", f"{files_count}/4")
        
        with c6:
            st.metric("📊 Total Rows", f"{total_rows:,}")
        
        with c7:
            st.metric("📋 Total Columns", f"{total_columns}")
        
        with c8:
            st.metric("🏪 Unique Stores", f"{len(all_stores)}")

    # Campaign & Feature Analysis with colourful badges
    st.markdown("### 🎯 Campaign & Feature Analysis")
    
    # Calculate campaign and feature metrics
    all_campaigns = set()
    for df in [marketing_df, ops_df, sales_df, payout_df]:
        campaign_cols = [col for col in df.columns if 'campaign' in col.lower()]
        for col in campaign_cols:
            all_campaigns.update(df[col].dropna().unique())
    
    important_features = set()
    key_metrics = [
        'orders', 'sales', 'revenue', 'commission', 'payout', 'rating', 
        'delivery', 'cancellation', 'downtime', 'promotion', 'ad', 'discount'
    ]
    for df in [marketing_df, ops_df, sales_df, payout_df]:
        for col in df.columns:
            col_lower = col.lower()
            if any(metric in col_lower for metric in key_metrics):
                important_features.add(col)
    
    total_cells = sum([
        df.size for df in [marketing_df, ops_df, sales_df, payout_df]
    ])
    non_null_cells = sum([
        df.count().sum() for df in [marketing_df, ops_df, sales_df, payout_df]
    ])
    completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
    
    all_dates = []
    for df in [marketing_df, ops_df, sales_df, payout_df]:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            all_dates.extend(dates)
    
    date_range_days = "N/A"
    if all_dates:
        date_range = max(all_dates) - min(all_dates)
        date_range_days = f"{date_range.days} days"
    
    # Display with colourful badges and metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📢 Unique Campaigns", f"{len(all_campaigns)}")
        if badge:
            badge(type="success", text=f"{len(all_campaigns)} campaigns active")
    
    with col2:
        st.metric("🔧 Key Features", f"{len(important_features)}")
        if badge:
            badge(type="info", text=f"{len(important_features)} key metrics tracked")
    
    with col3:
        st.metric("✅ Data Completeness", f"{completeness:.1f}%")
        if badge:
            badge(type="success" if completeness > 80 else "warning", 
                  text=f"{completeness:.1f}% complete")
    
    with col4:
        st.metric("📅 Date Range", date_range_days)
        if badge and date_range_days != "N/A":
            badge(type="primary", text=f"{date_range_days} coverage")

    # File-specific breakdown with detailed analysis
    st.markdown("### 📋 File Breakdown")
    st.markdown("*Detailed analysis of each dataset's characteristics*")
    file_data = []
    
    for name, df in [("Marketing", marketing_df), ("Operations", ops_df), ("Sales", sales_df), ("Payouts", payout_df)]:
        # Find date column for each file
        date_col = None
        if name == "Marketing" and "Date" in df.columns:
            date_col = "Date"
        elif name == "Operations" and "Start Date" in df.columns:
            date_col = "Start Date"
        elif name == "Sales" and "Start Date" in df.columns:
            date_col = "Start Date"
        elif name == "Payouts" and "Payout Date" in df.columns:
            date_col = "Payout Date"
        
        # Calculate date range
        date_range = "N/A"
        if date_col and not df.empty:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        
        # Count unique stores
        unique_stores = 0
        if "Store Name" in df.columns:
            unique_stores = len(df["Store Name"].dropna().unique())
        elif "Store ID" in df.columns:
            unique_stores = len(df["Store ID"].dropna().unique())
        
        # Count unique campaigns
        unique_campaigns = 0
        campaign_cols = [col for col in df.columns if 'campaign' in col.lower()]
        for col in campaign_cols:
            unique_campaigns += len(df[col].dropna().unique())
        
        file_data.append({
            "File": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Unique Stores": unique_stores,
            "Unique Campaigns": unique_campaigns,
            "Date Range": date_range
        })
    
    file_df = pd.DataFrame(file_data)
    st.dataframe(file_df, use_container_width=True)
    
    # Detailed column analysis by file
    st.markdown("### Important Columns by File")
    
    for name, df in [("Marketing", marketing_df), ("Operations", ops_df), ("Sales", sales_df), ("Payouts", payout_df)]:
        if not df.empty:
            st.markdown(f"**{name} File:**")
            
            # Find important columns
            important_cols = []
            key_metrics = [
                'orders', 'sales', 'revenue', 'commission', 'payout', 'rating', 
                'delivery', 'cancellation', 'downtime', 'promotion', 'ad', 'discount',
                'campaign', 'store', 'date', 'customer', 'fee', 'charge'
            ]
            
            for col in df.columns:
                col_lower = col.lower()
                if any(metric in col_lower for metric in key_metrics):
                    unique_count = len(df[col].dropna().unique())
                    important_cols.append({
                        "Column": col,
                        "Unique Values": unique_count,
                        "Data Type": str(df[col].dtype)
                    })
            
            if important_cols:
                col_df = pd.DataFrame(important_cols).sort_values("Unique Values", ascending=False)
                st.dataframe(col_df, use_container_width=True)
            else:
                st.info("No important columns found in this file.")
            
            st.markdown("---")
    
    # Enhanced Visualizations with colourful themes
    if px is not None:
        st.markdown("### 📊 Data Visualizations")
        st.markdown("*Interactive charts showing data distribution across files*")
        
        # Create a 2x2 grid for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # File sizes comparison with enhanced styling
            fig1 = px.bar(
                file_df, 
                x="File", 
                y="Rows", 
                title="📈 Number of Rows by File",
                color="File",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_dark"
            )
            fig1.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Unique stores by file
            if "Unique Stores" in file_df.columns and file_df["Unique Stores"].sum() > 0:
                fig3 = px.bar(
                    file_df, 
                    x="File", 
                    y="Unique Stores", 
                    title="🏪 Unique Stores by File",
                    color="File",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template="plotly_dark"
                )
                fig3.update_layout(
                    title_font_size=16,
                    title_font_color="#FF6B35",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Column count comparison with enhanced styling
            fig2 = px.bar(
                file_df, 
                x="File", 
                y="Columns", 
                title="📋 Number of Columns by File",
                color="File",
                color_discrete_sequence=px.colors.qualitative.Set1,
                template="plotly_dark"
            )
            fig2.update_layout(
                title_font_size=16,
                title_font_color="#FF6B35",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Unique campaigns by file
            if "Unique Campaigns" in file_df.columns and file_df["Unique Campaigns"].sum() > 0:
                fig4 = px.bar(
                    file_df, 
                    x="File", 
                    y="Unique Campaigns", 
                    title="📢 Unique Campaigns by File",
                    color="File",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    template="plotly_dark"
                )
                fig4.update_layout(
                    title_font_size=16,
                    title_font_color="#FF6B35",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        # Enhanced Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        st.markdown("*Overall data summary across all datasets*")
        
        summary_data = {
            "📁 Metric": ["Total Files", "Total Rows", "Total Columns", "Total Unique Stores", "Total Unique Campaigns"],
            "📈 Value": [
                files_count,
                total_rows,
                total_columns,
                len(all_stores),
                len(all_campaigns)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Style the dataframe with custom CSS
        st.markdown("""
        <style>
        .summary-table {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            summary_df, 
            use_container_width=True,
            hide_index=True
        )


# -------------------------------
# Main App
# -------------------------------

def main():
    st.set_page_config(
        page_title="🚀 DoorDash Performance Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .success-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        color: white !important;
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] label {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] div {
        color: white !important;
    }
    .stMetric div[data-testid="metric-container"] span {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Colourful header with icon
    if colored_header:
        colored_header(
            label="🚀 DoorDash Performance Dashboard",
            description="Comprehensive analytics and insights for DoorDash operations",
            color_name="orange-70"
        )
    else:
        st.markdown('<div class="main-header"><h1>🚀 DoorDash Performance Dashboard</h1><p>Comprehensive analytics and insights for DoorDash operations</p></div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("🚀 Loading DoorDash datasets..."):
        mk = load_csv(CSV_FILES["marketing"])  # marketing
        ops = load_csv(CSV_FILES["operations"])  # operations
        pay = load_csv(CSV_FILES["payouts"])  # payouts
        sal = load_csv(CSV_FILES["sales"])  # sales
    
    # Success message with animation
    if st_lottie:
        try:
            # Simple success animation
            st_lottie(
                "https://assets5.lottiefiles.com/packages/lf20_49rdyysj.json",
                height=100,
                key="success"
            )
        except:
            st.success("✅ All datasets loaded successfully!")
    else:
        st.success("✅ All datasets loaded successfully!")

    # Enhanced sidebar with clickable buttons
    with st.sidebar:
        st.header("📊 Dashboard Navigation")
        
        # Create buttons for navigation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏠 Overview", key="btn_overview", use_container_width=True):
                selected = "Overview"
            if st.button("📢 Marketing", key="btn_marketing", use_container_width=True):
                selected = "Marketing"
            if st.button("⚙️ Operations", key="btn_operations", use_container_width=True):
                selected = "Operations"
        
        with col2:
            if st.button("💰 Sales", key="btn_sales", use_container_width=True):
                selected = "Sales"
            if st.button("💸 Payouts", key="btn_payouts", use_container_width=True):
                selected = "Payouts"
        
        # Default selection if no button is clicked
        if 'selected' not in locals():
            selected = "Overview"

        st.markdown("---")
        st.header("🔍 Global Filters")
    # Determine min/max dates across datasets for convenience
    date_candidates: List[Tuple[pd.Series, str]] = []
    if "Date" in mk.columns:
        date_candidates.append((pd.to_datetime(mk["Date"], errors="coerce"), "Date"))
    if "Start Date" in ops.columns:
        date_candidates.append((pd.to_datetime(ops["Start Date"], errors="coerce"), "Start Date"))
    if "Payout Date" in pay.columns:
        date_candidates.append((pd.to_datetime(pay["Payout Date"], errors="coerce"), "Payout Date"))
    if "Start Date" in sal.columns:
        date_candidates.append((pd.to_datetime(sal["Start Date"], errors="coerce"), "Start Date"))

    all_dates = pd.concat([s for s, _ in date_candidates], axis=0) if date_candidates else pd.Series([], dtype="datetime64[ns]")
    min_date = pd.to_datetime(all_dates.min()) if not all_dates.empty else None
    max_date = pd.to_datetime(all_dates.max()) if not all_dates.empty else None

    date_filter = None
    if min_date is not None and max_date is not None:
        date_filter = st.sidebar.date_input("Date range", (min_date, max_date))

    if isinstance(date_filter, tuple) and len(date_filter) == 2:
        start_date = pd.to_datetime(date_filter[0])
        end_date = pd.to_datetime(date_filter[1])
        # Apply dataset-specific filters
        if "Date" in mk.columns:
            mk = filter_df_by_date(mk, "Date", start_date, end_date)
        if "Start Date" in ops.columns:
            ops = filter_df_by_date(ops, "Start Date", start_date, end_date)
        if "Payout Date" in pay.columns:
            pay = filter_df_by_date(pay, "Payout Date", start_date, end_date)
        if "Start Date" in sal.columns:
            sal = filter_df_by_date(sal, "Start Date", start_date, end_date)

    # Enhanced navigation based on selected option
    if selected == "Overview":
        section_overview(mk, ops, sal, pay)
    elif selected == "Marketing":
        section_marketing(mk)
    elif selected == "Operations":
        section_operations(ops)
    elif selected == "Sales":
        section_sales(sal)
    elif selected == "Payouts":
        section_payouts(pay)


if __name__ == "__main__":
    main()
