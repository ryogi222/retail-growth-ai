from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "daily_product_sales.csv"
TRANSACTION_FILE = PROJECT_ROOT / "data" / "retail_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading retail data...")

daily = pd.read_csv(DATA_FILE, parse_dates=["date"])
transactions = pd.read_csv(
    TRANSACTION_FILE,
    parse_dates=["date"],
)

# Product information
product_information = (
    transactions.groupby("product_id", as_index=False)
    .agg(
        product_name=("product_name", "first"),
        category=("category", "first"),
        base_price=("unit_price", "first"),
    )
)

dates = pd.date_range(
    daily["date"].min(),
    daily["date"].max(),
    freq="D",
)

stores = sorted(daily["store_id"].unique())
products = sorted(daily["product_id"].unique())

# Create every date-store-product combination.
complete_grid = pd.MultiIndex.from_product(
    [dates, stores, products],
    names=["date", "store_id", "product_id"],
).to_frame(index=False)

data = complete_grid.merge(
    daily.drop(columns=["product_name", "category"]),
    on=["date", "store_id", "product_id"],
    how="left",
)

data = data.merge(
    product_information,
    on="product_id",
    how="left",
)

zero_columns = [
    "units_sold",
    "revenue",
    "profit",
    "average_discount",
    "promotion",
    "transaction_count",
]

data[zero_columns] = data[zero_columns].fillna(0)

data["average_selling_price"] = data[
    "average_selling_price"
].fillna(data["base_price"])

# Calendar features
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_month"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
data["day_of_year"] = data["date"].dt.dayofyear
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)

data["is_holiday_period"] = (
    data["month"].isin([11, 12])
).astype(int)

# Sort before generating historical features.
data = data.sort_values(
    ["store_id", "product_id", "date"]
).reset_index(drop=True)

grouped_sales = data.groupby(
    ["store_id", "product_id"]
)["units_sold"]

# Historical demand features
data["sales_lag_1"] = grouped_sales.shift(1)
data["sales_lag_7"] = grouped_sales.shift(7)
data["sales_lag_14"] = grouped_sales.shift(14)
data["sales_lag_28"] = grouped_sales.shift(28)

data["rolling_mean_7"] = grouped_sales.transform(
    lambda values: values.shift(1).rolling(7).mean()
)

data["rolling_mean_28"] = grouped_sales.transform(
    lambda values: values.shift(1).rolling(28).mean()
)

data["rolling_std_7"] = grouped_sales.transform(
    lambda values: values.shift(1).rolling(7).std()
)

# Remove rows without sufficient history.
model_data = data.dropna(
    subset=[
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
    ]
).copy()

# Convert categories into numeric indicator columns.
model_data = pd.get_dummies(
    model_data,
    columns=["store_id", "product_id", "category"],
    dtype=int,
)

excluded_columns = [
    "date",
    "product_name",
    "units_sold",
    "revenue",
    "profit",
]

feature_columns = [
    column
    for column in model_data.columns
    if column not in excluded_columns
]

# Time-based split: final 60 days are unseen test data.
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
print(f"Test period begins: {test_start_date.date()}")
print("Training Demand Forecast Agent...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
r2 = r2_score(y_test, predictions)

wmape = (
    np.abs(y_test.to_numpy() - predictions).sum()
    / max(y_test.sum(), 1)
    * 100
)

baseline_predictions = X_test["sales_lag_7"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

improvement = (
    (baseline_mae - mae) / baseline_mae * 100
    if baseline_mae > 0
    else 0
)

print("\nDEMAND FORECAST RESULTS")
print(f"MAE: {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")
print(f"R²: {r2:.3f}")
print(f"WMAPE: {wmape:.2f}%")
print(f"Seven-day baseline MAE: {baseline_mae:.3f}")
print(f"Improvement over baseline: {improvement:.2f}%")

# Save model and information required for future predictions.
model_bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "test_start_date": str(test_start_date.date()),
    "metrics": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "wmape": wmape,
        "baseline_mae": baseline_mae,
        "baseline_improvement_pct": improvement,
    },
}

model_file = MODEL_DIR / "demand_forecast_model.joblib"
joblib.dump(model_bundle, model_file)

# Save test predictions.
prediction_report = test_data[
    ["date", "product_name", "units_sold"]
].copy()

prediction_report["predicted_units"] = predictions
prediction_report["absolute_error"] = np.abs(
    prediction_report["units_sold"]
    - prediction_report["predicted_units"]
)

prediction_report.to_csv(
    REPORT_DIR / "demand_forecast_predictions.csv",
    index=False,
)

# Save feature importance.
importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

importance.to_csv(
    REPORT_DIR / "demand_feature_importance.csv",
    index=False,
)

print(f"\nModel saved to: {model_file}")
print("Demand Forecast Agent completed successfully")
