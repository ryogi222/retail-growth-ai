from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_ARTIFACTS = [
    PROJECT_ROOT / "data" / "retail_transactions.csv",
    PROJECT_ROOT / "data" / "daily_product_sales.csv",
    PROJECT_ROOT / "models" / "demand_forecast_model.joblib",
    PROJECT_ROOT / "models" / "promotion_profit_model.joblib",
    PROJECT_ROOT / "models" / "basket_recommendation_model.joblib",
    PROJECT_ROOT / "reports" / "demand_forecast_predictions.csv",
    PROJECT_ROOT / "reports" / "promotion_recommendations.csv",
    PROJECT_ROOT / "reports" / "basket_recommendations.csv",
    PROJECT_ROOT / "reports" / "executive_decisions.csv",
    PROJECT_ROOT / "reports" / "executive_summary.json",
]


def run_script(script_path):
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Startup script failed: {script_path.name}\n"
            f"{result.stderr[-3000:]}"
        )


def ensure_project_artifacts():
    missing_files = [
        file
        for file in REQUIRED_ARTIFACTS
        if not file.exists()
    ]

    if not missing_files:
        return

    transaction_file = (
        PROJECT_ROOT
        / "data"
        / "retail_transactions.csv"
    )

    if not transaction_file.exists():
        run_script(
            PROJECT_ROOT
            / "data"
            / "generate_retail_data.py"
        )

    run_script(PROJECT_ROOT / "run_pipeline.py")

    remaining_missing_files = [
        file
        for file in REQUIRED_ARTIFACTS
        if not file.exists()
    ]

    if remaining_missing_files:
        missing_names = "\n".join(
            str(file)
            for file in remaining_missing_files
        )

        raise RuntimeError(
            "The following startup artifacts are missing:\n"
            + missing_names
        )