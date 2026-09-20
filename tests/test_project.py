from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def test_required_data_files_exist():
    required_files = [
        DATA_DIR / "retail_transactions.csv",
        DATA_DIR / "daily_product_sales.csv",
    ]

    for file in required_files:
        assert file.exists(), f"Missing data file: {file}"


def test_transaction_columns_exist():
    transactions = pd.read_csv(
        DATA_DIR / "retail_transactions.csv"
    )

    expected_columns = {
        "transaction_id",
        "date",
        "store_id",
        "customer_id",
        "product_id",
        "product_name",
        "category",
        "quantity",
        "unit_price",
        "unit_cost",
        "promotion",
        "discount_pct",
        "selling_price",
        "revenue",
        "profit",
    }

    assert expected_columns.issubset(
        transactions.columns
    )


def test_transaction_data_quality():
    transactions = pd.read_csv(
        DATA_DIR / "retail_transactions.csv"
    )

    assert not transactions.empty
    assert transactions.isna().sum().sum() == 0
    assert (transactions["quantity"] > 0).all()
    assert (transactions["selling_price"] > 0).all()

    assert transactions["discount_pct"].between(
        0,
        1,
    ).all()


def test_revenue_calculation():
    transactions = pd.read_csv(
        DATA_DIR / "retail_transactions.csv"
    )

    expected_revenue = (
        transactions["selling_price"]
        * transactions["quantity"]
    ).round(2)

    assert np.allclose(
        transactions["revenue"],
        expected_revenue,
        atol=0.01,
    )


def test_profit_calculation():
    transactions = pd.read_csv(
        DATA_DIR / "retail_transactions.csv"
    )

    expected_profit = (
        (
            transactions["selling_price"]
            - transactions["unit_cost"]
        )
        * transactions["quantity"]
    ).round(2)

    assert np.allclose(
        transactions["profit"],
        expected_profit,
        atol=0.01,
    )


def test_daily_model_data():
    daily = pd.read_csv(
        DATA_DIR / "daily_product_sales.csv"
    )

    expected_columns = {
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "revenue",
        "profit",
        "promotion",
    }

    assert expected_columns.issubset(daily.columns)
    assert not daily.empty
    assert (daily["units_sold"] >= 0).all()


def test_model_files_load_successfully():
    model_files = [
        MODEL_DIR / "demand_forecast_model.joblib",
        MODEL_DIR / "promotion_profit_model.joblib",
        MODEL_DIR / "basket_recommendation_model.joblib",
    ]

    for model_file in model_files:
        assert model_file.exists(), (
            f"Missing model: {model_file}"
        )

        model_bundle = joblib.load(model_file)

        assert model_bundle is not None


def test_demand_forecast_output():
    forecasts = pd.read_csv(
        REPORT_DIR / "demand_forecast_predictions.csv"
    )

    assert not forecasts.empty
    assert {
        "units_sold",
        "predicted_units",
        "absolute_error",
    }.issubset(forecasts.columns)

    assert (forecasts["predicted_units"] >= 0).all()
    assert (forecasts["absolute_error"] >= 0).all()


def test_promotion_recommendations():
    promotions = pd.read_csv(
        REPORT_DIR / "promotion_recommendations.csv"
    )

    assert not promotions.empty

    required_columns = {
        "recommendation",
        "expected_profit",
        "baseline_expected_profit",
        "incremental_profit",
    }

    assert required_columns.issubset(
        promotions.columns
    )

    # The optimiser includes no discount as an option,
    # so its selected profit should not be below baseline.
    assert (
        promotions["expected_profit"]
        + 0.01
        >= promotions["baseline_expected_profit"]
    ).all()


def test_basket_recommendations():
    baskets = pd.read_csv(
        REPORT_DIR / "basket_recommendations.csv"
    )

    assert not baskets.empty
    assert (baskets["lift"] >= 1.0).all()
    assert baskets["confidence"].between(
        0,
        100,
    ).all()

    assert (
        baskets["source_product"]
        != baskets["recommended_product"]
    ).all()


def test_supervisor_decisions():
    decisions = pd.read_csv(
        REPORT_DIR / "executive_decisions.csv"
    )

    assert not decisions.empty

    required_columns = {
        "decision_id",
        "priority",
        "agent_source",
        "action",
        "reason",
        "expected_value_gbp",
        "confidence_score",
        "status",
    }

    assert required_columns.issubset(
        decisions.columns
    )

    assert set(decisions["priority"]).issubset(
        {"High", "Medium", "Low"}
    )

    assert decisions["confidence_score"].between(
        0,
        100,
    ).all()

    assert (
        decisions["status"]
        == "Pending Manager Review"
    ).all()


def test_pipeline_status():
    log_file = REPORT_DIR / "pipeline_run_log.json"

    assert log_file.exists()

    with open(
        log_file,
        "r",
        encoding="utf-8",
    ) as file:
        pipeline_log = json.load(file)

    assert pipeline_log["pipeline_status"] == "SUCCESS"
    assert (
        pipeline_log["successful_steps"]
        == pipeline_log["total_steps"]
    )