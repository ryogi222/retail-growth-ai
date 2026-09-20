from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

FORECAST_FILE = REPORT_DIR / "demand_forecast_predictions.csv"
PROMOTION_FILE = REPORT_DIR / "promotion_recommendations.csv"
BASKET_FILE = REPORT_DIR / "basket_recommendations.csv"

OUTPUT_FILE = REPORT_DIR / "executive_decisions.csv"
SUMMARY_FILE = REPORT_DIR / "executive_summary.json"


def normalise_score(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.full(len(series), 50.0),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


def assign_priority(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


print("Supervisor Agent is reviewing specialist outputs...")

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["date"],
)

promotions = pd.read_csv(
    PROMOTION_FILE,
    parse_dates=["date"],
)

baskets = pd.read_csv(BASKET_FILE)

decisions = []

# -------------------------------------------------
# 1. Promotion Agent decisions
# -------------------------------------------------
promotion_actions = promotions[
    promotions["incremental_profit"] > 0
].copy()

promotion_actions["score"] = normalise_score(
    promotion_actions["incremental_profit"]
)

promotion_actions = promotion_actions.nlargest(
    10,
    "score",
)

for _, row in promotion_actions.iterrows():
    decisions.append(
        {
            "agent_source": "Promotion Profit Agent",
            "store_id": row["store_id"],
            "product": row["product_name"],
            "action": row["recommendation"],
            "reason": (
                "This option produced the highest expected "
                "profit across the tested discount levels."
            ),
            "expected_value_gbp": round(
                row["incremental_profit"],
                2,
            ),
            "confidence_score": round(row["score"], 1),
        }
    )

# -------------------------------------------------
# 2. Basket Agent decisions
# -------------------------------------------------
basket_actions = baskets.copy()

basket_actions["score"] = normalise_score(
    basket_actions["opportunity_score"]
)

basket_actions = basket_actions.nlargest(
    10,
    "score",
)

for _, row in basket_actions.iterrows():
    decisions.append(
        {
            "agent_source": "Basket Recommendation Agent",
            "store_id": "All Stores",
            "product": row["source_product"],
            "action": (
                f"Offer {row['recommended_product']} "
                "as a 10% discounted add-on"
            ),
            "reason": (
                f"Basket confidence is "
                f"{row['confidence']:.1f}% with "
                f"a lift of {row['lift']:.2f}."
            ),
            "expected_value_gbp": round(
                row["estimated_bundle_profit"],
                2,
            ),
            "confidence_score": round(row["score"], 1),
        }
    )

# -------------------------------------------------
# 3. Demand Forecast Agent decisions
# -------------------------------------------------
demand_actions = (
    forecast.groupby("product_name", as_index=False)
    .agg(
        actual_units=("units_sold", "sum"),
        predicted_units=("predicted_units", "sum"),
    )
)

demand_actions["demand_gap"] = (
    demand_actions["predicted_units"]
    - demand_actions["actual_units"]
)

# Products where predicted demand exceeds actual demand.
demand_actions = demand_actions[
    demand_actions["demand_gap"] > 0
].copy()

if not demand_actions.empty:
    demand_actions["score"] = normalise_score(
        demand_actions["demand_gap"]
    )

    demand_actions = demand_actions.nlargest(
        10,
        "score",
    )

    for _, row in demand_actions.iterrows():
        percentage_increase = (
            row["demand_gap"]
            / max(row["actual_units"], 1)
            * 100
        )

        decisions.append(
            {
                "agent_source": "Demand Forecast Agent",
                "store_id": "All Stores",
                "product": row["product_name"],
                "action": (
                    "Review availability and increase "
                    "replenishment if operationally appropriate"
                ),
                "reason": (
                    f"Predicted demand is "
                    f"{percentage_increase:.1f}% above "
                    "the comparison demand level."
                ),
                "expected_value_gbp": 0.0,
                "confidence_score": round(
                    row["score"],
                    1,
                ),
            }
        )

decision_table = pd.DataFrame(decisions)

if decision_table.empty:
    raise ValueError(
        "The Supervisor Agent found no actionable decisions."
    )

decision_table["priority"] = decision_table[
    "confidence_score"
].apply(assign_priority)

priority_order = {
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

decision_table["priority_order"] = decision_table[
    "priority"
].map(priority_order)

decision_table = decision_table.sort_values(
    [
        "priority_order",
        "expected_value_gbp",
        "confidence_score",
    ],
    ascending=[True, False, False],
).reset_index(drop=True)

decision_table.insert(
    0,
    "decision_id",
    [
        f"DEC-{number:03d}"
        for number in range(
            1,
            len(decision_table) + 1,
        )
    ],
)

decision_table["status"] = "Pending Manager Review"

decision_table = decision_table.drop(
    columns=["priority_order"]
)

decision_table.to_csv(
    OUTPUT_FILE,
    index=False,
)

high_priority_count = int(
    (decision_table["priority"] == "High").sum()
)

potential_value = float(
    decision_table["expected_value_gbp"].sum()
)

executive_summary = {
    "generated_at": pd.Timestamp.now(
        tz="UTC"
    ).isoformat(),
    "total_decisions": len(decision_table),
    "high_priority_decisions": high_priority_count,
    "potential_value_gbp": round(
        potential_value,
        2,
    ),
    "agents_reviewed": [
        "Demand Forecast Agent",
        "Promotion Profit Agent",
        "Basket Recommendation Agent",
    ],
    "status": "Pending Manager Review",
    "data_type": "Synthetic retail prototype",
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        executive_summary,
        file,
        indent=4,
    )

print("\nSUPERVISOR AGENT RESULTS")
print(f"Total decisions: {len(decision_table)}")
print(
    f"High-priority decisions: {high_priority_count}"
)
print(
    f"Potential opportunity value: "
    f"£{potential_value:,.2f}"
)

print("\nTop management decisions:")
print(
    decision_table[
        [
            "decision_id",
            "priority",
            "agent_source",
            "product",
            "action",
            "expected_value_gbp",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Saved: {SUMMARY_FILE}")
print("Supervisor Agent completed successfully")
