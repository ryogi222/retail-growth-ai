from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = PROJECT_ROOT / "data" / "daily_product_sales.csv"
TRANSACTION_FILE = PROJECT_ROOT / "data" / "retail_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading promotion and sales data...")

daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])
transactions = pd.read_csv(
    TRANSACTION_FILE,
    parse_dates=["date"],
)

product_information = (
    transactions.groupby("product_id", as_index=False)
    .agg(
        product_name=("product_name", "first"),
        category=("category", "first"),
        base_price=("unit_price", "first"),
        unit_cost=("unit_cost", "first"),
    )
)

data = daily.drop(
    columns=["product_name", "category"]
).merge(
    product_information,
    on="product_id",
    how="left",
)

# Calendar features
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["week_of_year"] = (
    data["date"].dt.isocalendar().week.astype(int)
)
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)
data["is_holiday_period"] = (
    data["month"].isin([11, 12])
).astype(int)

# Create model-ready categorical columns.
model_data = pd.get_dummies(
    data,
    columns=["store_id", "product_id", "category"],
    dtype=int,
)

excluded_columns = [
    "date",
    "product_name",
    "units_sold",
    "revenue",
    "profit",
    "transaction_count",
]

feature_columns = [
    column
    for column in model_data.columns
    if column not in excluded_columns
]

# Use the final 60 days as unseen test data.
test_start_date = (
    model_data["date"].max()
    - pd.Timedelta(days=59)
)

train_data = model_data[
    model_data["date"] < test_start_date
].copy()

test_data = model_data[
    model_data["date"] >= test_start_date
].copy()

X_train = train_data[feature_columns]
y_train = train_data["units_sold"]

X_test = test_data[feature_columns]
y_test = test_data["units_sold"]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print("Training Promotion Agent...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=350,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = np.maximum(model.predict(X_test), 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\nPROMOTION MODEL RESULTS")
print(f"MAE: {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")

# Create scenarios for the next trading day.
next_date = data["date"].max() + pd.Timedelta(days=1)
discount_options = [0.00, 0.10, 0.15, 0.20, 0.25]

scenario_rows = []

store_ids = sorted(data["store_id"].unique())

for store_id in store_ids:
    for _, product in product_information.iterrows():
        for discount in discount_options:
            selling_price = round(
                product["base_price"] * (1 - discount),
                2,
            )

            scenario_rows.append(
                {
                    "date": next_date,
                    "store_id": store_id,
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "average_selling_price": selling_price,
                    "average_discount": discount,
                    "promotion": int(discount > 0),
                    "base_price": product["base_price"],
                    "unit_cost": product["unit_cost"],
                    "day_of_week": next_date.dayofweek,
                    "month": next_date.month,
                    "week_of_year": int(
                        next_date.isocalendar().week
                    ),
                    "is_weekend": int(
                        next_date.dayofweek >= 5
                    ),
                    "is_holiday_period": int(
                        next_date.month in [11, 12]
                    ),
                }
            )

scenarios = pd.DataFrame(scenario_rows)

scenario_model_data = pd.get_dummies(
    scenarios,
    columns=["store_id", "product_id", "category"],
    dtype=int,
)

scenario_features = scenario_model_data.reindex(
    columns=feature_columns,
    fill_value=0,
)

scenarios["predicted_units"] = np.maximum(
    model.predict(scenario_features),
    0,
)

scenarios["expected_revenue"] = (
    scenarios["predicted_units"]
    * scenarios["average_selling_price"]
)

scenarios["expected_profit"] = (
    scenarios["predicted_units"]
    * (
        scenarios["average_selling_price"]
        - scenarios["unit_cost"]
    )
)

# Baseline profit with no discount.
baseline = (
    scenarios[scenarios["average_discount"] == 0]
    [
        ["store_id", "product_id", "expected_profit"]
    ]
    .rename(
        columns={
            "expected_profit": "baseline_expected_profit"
        }
    )
)

scenarios = scenarios.merge(
    baseline,
    on=["store_id", "product_id"],
    how="left",
)

scenarios["incremental_profit"] = (
    scenarios["expected_profit"]
    - scenarios["baseline_expected_profit"]
)

# Choose the most profitable scenario for each store and product.
best_scenario_indexes = scenarios.groupby(
    ["store_id", "product_id"]
)["expected_profit"].idxmax()

recommendations = scenarios.loc[
    best_scenario_indexes
].copy()

recommendations["recommendation"] = np.where(
    recommendations["average_discount"] > 0,
    (
        "Apply "
        + (
            recommendations["average_discount"] * 100
        ).round(0).astype(int).astype(str)
        + "% discount"
    ),
    "No discount",
)

recommendations = recommendations[
    [
        "date",
        "store_id",
        "product_id",
        "product_name",
        "category",
        "base_price",
        "unit_cost",
        "average_discount",
        "recommendation",
        "predicted_units",
        "expected_revenue",
        "expected_profit",
        "baseline_expected_profit",
        "incremental_profit",
    ]
].sort_values(
    ["incremental_profit", "expected_profit"],
    ascending=False,
)

currency_columns = [
    "base_price",
    "unit_cost",
    "expected_revenue",
    "expected_profit",
    "baseline_expected_profit",
    "incremental_profit",
]

recommendations[currency_columns] = recommendations[
    currency_columns
].round(2)

recommendations["predicted_units"] = recommendations[
    "predicted_units"
].round(1)

recommendations.to_csv(
    REPORT_DIR / "promotion_recommendations.csv",
    index=False,
)

model_bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "metrics": {
        "mae": mae,
        "rmse": rmse,
    },
    "discount_options": discount_options,
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "promotion_profit_model.joblib",
)

total_baseline_profit = recommendations[
    "baseline_expected_profit"
].sum()

total_recommended_profit = recommendations[
    "expected_profit"
].sum()

profit_uplift = (
    (
        total_recommended_profit
        - total_baseline_profit
    )
    / max(total_baseline_profit, 1)
    * 100
)

print("\nPROMOTION RECOMMENDATIONS")
print(f"Recommendation date: {next_date.date()}")
print(
    "Products recommended for promotion: "
    f"{(recommendations['average_discount'] > 0).sum()}"
)
print(
    "Expected baseline profit: "
    f"£{total_baseline_profit:,.2f}"
)
print(
    "Expected optimised profit: "
    f"£{total_recommended_profit:,.2f}"
)
print(f"Expected profit uplift: {profit_uplift:.2f}%")

print("\nTop recommendations:")
print(
    recommendations[
        [
            "store_id",
            "product_name",
            "recommendation",
            "predicted_units",
            "expected_profit",
            "incremental_profit",
        ]
    ].head(10).to_string(index=False)
)

print(
    "\nSaved: reports/promotion_recommendations.csv"
)
print(
    "Saved: models/promotion_profit_model.joblib"
)
print("Promotion and Profit Agent completed successfully")
