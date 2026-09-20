from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

DECISION_FILE = REPORT_DIR / "executive_decisions.csv"
SUMMARY_FILE = REPORT_DIR / "executive_summary.json"

st.set_page_config(
    page_title="Executive Decisions",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-left: 5px solid #00539F;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .decision-banner {
        background-color: #FFFFFF;
        border-left: 6px solid #E31837;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DECISION_FILE.exists() or not SUMMARY_FILE.exists():
    st.error(
        "Supervisor outputs are missing. Run "
        "`python agents\\supervisor_agent.py` first."
    )
    st.stop()

decisions = pd.read_csv(DECISION_FILE)

with open(
    SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:
    summary = json.load(file)

st.title("🧠 Executive Decision Centre")

st.markdown(
    """
    <div class="decision-banner">
        The Supervisor Agent combines demand forecasts,
        promotion recommendations and basket opportunities
        into one prioritised management action list.
        All recommendations require manager review.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Total Decisions",
    summary["total_decisions"],
)

metric_2.metric(
    "High Priority",
    summary["high_priority_decisions"],
)

metric_3.metric(
    "Potential Value",
    f"£{summary['potential_value_gbp']:,.2f}",
)

metric_4.metric(
    "Agents Reviewed",
    len(summary["agents_reviewed"]),
)

st.subheader("Management filters")

filter_1, filter_2, filter_3 = st.columns(3)

with filter_1:
    selected_priority = st.selectbox(
        "Priority",
        ["All"] + sorted(decisions["priority"].unique()),
    )

with filter_2:
    selected_agent = st.selectbox(
        "Specialist Agent",
        ["All"] + sorted(decisions["agent_source"].unique()),
    )

with filter_3:
    selected_store = st.selectbox(
        "Store",
        ["All"] + sorted(decisions["store_id"].unique()),
    )

filtered = decisions.copy()

if selected_priority != "All":
    filtered = filtered[
        filtered["priority"] == selected_priority
    ]

if selected_agent != "All":
    filtered = filtered[
        filtered["agent_source"] == selected_agent
    ]

if selected_store != "All":
    filtered = filtered[
        filtered["store_id"] == selected_store
    ]

st.subheader("Prioritised opportunity value")

chart_data = (
    filtered.sort_values(
        "expected_value_gbp",
        ascending=False,
    )
    .head(15)
    .sort_values("expected_value_gbp")
)

if not chart_data.empty:
    opportunity_chart = px.bar(
        chart_data,
        x="expected_value_gbp",
        y="product",
        color="priority",
        orientation="h",
        hover_data=[
            "agent_source",
            "store_id",
            "action",
        ],
        color_discrete_map={
            "High": "#E31837",
            "Medium": "#F59E0B",
            "Low": "#00539F",
        },
        title="Top Management Opportunities",
        template="plotly_white",
    )

    opportunity_chart.update_layout(
        xaxis_title="Potential Value (£)",
        yaxis_title="",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1F2937"),
    )

    st.plotly_chart(
        opportunity_chart,
        use_container_width=True,
    )
else:
    st.info("No decisions match the selected filters.")

st.subheader("Management decision queue")

display_columns = [
    "decision_id",
    "priority",
    "agent_source",
    "store_id",
    "product",
    "action",
    "reason",
    "expected_value_gbp",
    "confidence_score",
    "status",
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "decision_id": "Decision ID",
        "priority": "Priority",
        "agent_source": "Source Agent",
        "store_id": "Store",
        "product": "Product",
        "action": "Recommended Action",
        "reason": "Business Reason",
        "expected_value_gbp": st.column_config.NumberColumn(
            "Potential Value",
            format="£%.2f",
        ),
        "confidence_score": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0,
            max_value=100,
            format="%.1f",
        ),
        "status": "Governance Status",
    },
)

csv_download = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Management Decisions",
    data=csv_download,
    file_name="retailgrowth_management_decisions.csv",
    mime="text/csv",
)

st.divider()

st.warning(
    "Prototype governance: recommendations must be reviewed "
    "and tested through a controlled store pilot before implementation."
)