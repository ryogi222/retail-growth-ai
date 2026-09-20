from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="RetailGrowth AI",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .main-title {
        color: #00539F;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #555555;
        font-size: 17px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border-left: 5px solid #00539F;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    .recommendation-box {
        background-color: #ffffff;
        border-left: 5px solid #E31837;
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    transactions = pd.read_csv(
        DATA_DIR / "retail_transactions.csv",
        parse_dates=["date"],
    )

    forecasts = pd.read_csv(
        REPORT_DIR / "demand_forecast_predictions.csv",
        parse_dates=["date"],
    )

    promotions = pd.read_csv(
        REPORT_DIR / "promotion_recommendations.csv",
        parse_dates=["date"],
    )

    baskets = pd.read_csv(
        REPORT_DIR / "basket_recommendations.csv"
    )

    return transactions, forecasts, promotions, baskets


@st.cache_resource
def load_model_metrics():
    forecast_bundle = joblib.load(
        MODEL_DIR / "demand_forecast_model.joblib"
    )

    promotion_bundle = joblib.load(
        MODEL_DIR / "promotion_profit_model.joblib"
    )

    return (
        forecast_bundle["metrics"],
        promotion_bundle["metrics"],
    )


required_files = [
    DATA_DIR / "retail_transactions.csv",
    REPORT_DIR / "demand_forecast_predictions.csv",
    REPORT_DIR / "promotion_recommendations.csv",
    REPORT_DIR / "basket_recommendations.csv",
    MODEL_DIR / "demand_forecast_model.joblib",
    MODEL_DIR / "promotion_profit_model.joblib",
]

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:
    st.error("Some required project files are missing.")
    st.code("\n".join(missing_files))
    st.stop()

transactions, forecasts, promotions, baskets = load_data()
forecast_metrics, promotion_metrics = load_model_metrics()

st.markdown(
    '<p class="main-title">RetailGrowth AI</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="subtitle">
    Multi-agent sales, promotion and basket-profit optimisation platform
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard filters")

    selected_store = st.selectbox(
        "Store",
        ["All Stores"]
        + sorted(transactions["store_id"].unique()),
    )

    selected_category = st.selectbox(
        "Category",
        ["All Categories"]
        + sorted(transactions["category"].unique()),
    )

    st.divider()

    st.success(
        "Demand Forecast Agent: Active"
    )
    st.success(
        "Promotion Profit Agent: Active"
    )
    st.success(
        "Basket Recommendation Agent: Active"
    )

filtered_data = transactions.copy()

if selected_store != "All Stores":
    filtered_data = filtered_data[
        filtered_data["store_id"] == selected_store
    ]

if selected_category != "All Categories":
    filtered_data = filtered_data[
        filtered_data["category"] == selected_category
    ]

total_revenue = filtered_data["revenue"].sum()
total_profit = filtered_data["profit"].sum()
total_transactions = filtered_data[
    "transaction_id"
].nunique()

average_basket = (
    total_revenue / total_transactions
    if total_transactions
    else 0
)

profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue
    else 0
)

tab_overview, tab_forecast, tab_promotion, tab_basket = st.tabs(
    [
        "Business Overview",
        "Demand Forecast",
        "Promotion Optimiser",
        "Basket Recommendations",
    ]
)

with tab_overview:
    st.subheader("Business performance")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Revenue",
        f"£{total_revenue:,.0f}",
    )

    metric_2.metric(
        "Profit",
        f"£{total_profit:,.0f}",
    )

    metric_3.metric(
        "Profit Margin",
        f"{profit_margin:.1f}%",
    )

    metric_4.metric(
        "Average Basket",
        f"£{average_basket:,.2f}",
    )

    monthly = (
        filtered_data.assign(
            month=filtered_data["date"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("month", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        )
    )

    monthly_long = monthly.melt(
        id_vars="month",
        value_vars=["revenue", "profit"],
        var_name="measure",
        value_name="amount",
    )

    chart_1, chart_2 = st.columns(2)

    with chart_1:
        monthly_chart = px.line(
            monthly_long,
            x="month",
            y="amount",
            color="measure",
            markers=True,
            title="Monthly Revenue and Profit",
            color_discrete_map={
                "revenue": "#00539F",
                "profit": "#E31837",
            },
        )

        monthly_chart.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            legend_title="",
        )

        st.plotly_chart(
            monthly_chart,
            use_container_width=True,
        )

    product_performance = (
        filtered_data.groupby(
            "product_name",
            as_index=False,
        )
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    with chart_2:
        product_chart = px.bar(
            product_performance.sort_values("revenue"),
            x="revenue",
            y="product_name",
            orientation="h",
            title="Top Products by Revenue",
            color_discrete_sequence=["#00539F"],
        )

        product_chart.update_layout(
            xaxis_title="Revenue (£)",
            yaxis_title="",
        )

        st.plotly_chart(
            product_chart,
            use_container_width=True,
        )

with tab_forecast:
    st.subheader("Demand Forecast Agent")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "MAE",
        f"{forecast_metrics['mae']:.2f} units",
    )

    metric_2.metric(
        "RMSE",
        f"{forecast_metrics['rmse']:.2f} units",
    )

    metric_3.metric(
        "WMAPE",
        f"{forecast_metrics['wmape']:.1f}%",
    )

    metric_4.metric(
        "Baseline Improvement",
        f"{forecast_metrics['baseline_improvement_pct']:.1f}%",
    )

    selected_product = st.selectbox(
        "Select a product",
        sorted(forecasts["product_name"].unique()),
        key="forecast_product",
    )

    product_forecast = forecasts[
        forecasts["product_name"] == selected_product
    ]

    forecast_by_date = (
        product_forecast.groupby("date", as_index=False)
        .agg(
            actual_units=("units_sold", "sum"),
            predicted_units=("predicted_units", "sum"),
        )
    )

    forecast_chart = go.Figure()

    forecast_chart.add_trace(
        go.Scatter(
            x=forecast_by_date["date"],
            y=forecast_by_date["actual_units"],
            name="Actual",
            mode="lines",
            line=dict(color="#00539F"),
        )
    )

    forecast_chart.add_trace(
        go.Scatter(
            x=forecast_by_date["date"],
            y=forecast_by_date["predicted_units"],
            name="Predicted",
            mode="lines",
            line=dict(color="#E31837", dash="dash"),
        )
    )

    forecast_chart.update_layout(
        title=f"Actual vs Predicted Demand: {selected_product}",
        xaxis_title="Date",
        yaxis_title="Units",
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True,
    )

    st.dataframe(
        forecast_by_date.sort_values(
            "date",
            ascending=False,
        ).head(20),
        use_container_width=True,
        hide_index=True,
    )

with tab_promotion:
    st.subheader("Promotion and Profit Optimisation Agent")

    optimised_profit = promotions[
        "expected_profit"
    ].sum()

    baseline_profit = promotions[
        "baseline_expected_profit"
    ].sum()

    incremental_profit = promotions[
        "incremental_profit"
    ].sum()

    profit_uplift = (
        incremental_profit / baseline_profit * 100
        if baseline_profit
        else 0
    )

    promoted_products = (
        promotions["average_discount"] > 0
    ).sum()

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Baseline Profit",
        f"£{baseline_profit:,.2f}",
    )

    metric_2.metric(
        "Optimised Profit",
        f"£{optimised_profit:,.2f}",
        delta=f"£{incremental_profit:,.2f}",
    )

    metric_3.metric(
        "Expected Profit Uplift",
        f"{profit_uplift:.2f}%",
    )

    metric_4.metric(
        "Promotions Recommended",
        int(promoted_products),
    )

    promotion_view = promotions[
        promotions["incremental_profit"] > 0
    ].copy()

    promotion_chart = px.bar(
        promotion_view.head(15).sort_values(
            "incremental_profit"
        ),
        x="incremental_profit",
        y="product_name",
        color="store_id",
        orientation="h",
        title="Top Expected Incremental-Profit Opportunities",
    )

    promotion_chart.update_layout(
        xaxis_title="Expected Incremental Profit (£)",
        yaxis_title="",
    )

    st.plotly_chart(
        promotion_chart,
        use_container_width=True,
    )

    st.dataframe(
        promotion_view[
            [
                "store_id",
                "product_name",
                "recommendation",
                "predicted_units",
                "expected_profit",
                "incremental_profit",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_basket:
    st.subheader("Basket Recommendation Agent")

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Product Relationships",
        len(baskets),
    )

    metric_2.metric(
        "Highest Lift",
        f"{baskets['lift'].max():.2f}",
    )

    metric_3.metric(
        "Estimated Bundle Profit",
        f"£{baskets['estimated_bundle_profit'].sum():,.2f}",
    )

    selected_source_product = st.selectbox(
        "Customer is purchasing",
        sorted(baskets["source_product"].unique()),
    )

    product_recommendations = baskets[
        baskets["source_product"]
        == selected_source_product
    ].sort_values(
        "opportunity_score",
        ascending=False,
    )

    if not product_recommendations.empty:
        best = product_recommendations.iloc[0]

        st.markdown(
            f"""
            <div class="recommendation-box">
                <b>Recommended add-on:</b>
                {best['recommended_product']}<br>
                <b>Offer:</b> 10% bundle discount<br>
                <b>Confidence:</b>
                {best['confidence']:.1f}%<br>
                <b>Lift:</b> {best['lift']:.2f}<br>
                <b>Profit per add-on:</b>
                £{best['profit_per_add_on']:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        product_recommendations[
            [
                "source_product",
                "recommended_product",
                "support",
                "confidence",
                "lift",
                "profit_per_add_on",
                "opportunity_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.caption(
    "RetailGrowth AI prototype — built using synthetic retail data. "
    "Recommendations require controlled business testing before deployment."
)