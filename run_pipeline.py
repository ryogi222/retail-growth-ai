from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

PIPELINE_STEPS = [
    {
        "name": "Data Validation",
        "script": PROJECT_ROOT
        / "notebooks"
        / "01_data_validation.py",
    },
    {
        "name": "Exploratory Analysis",
        "script": PROJECT_ROOT
        / "notebooks"
        / "02_exploratory_analysis.py",
    },
    {
        "name": "Demand Forecast Agent",
        "script": PROJECT_ROOT
        / "agents"
        / "demand_forecast_agent.py",
    },
    {
        "name": "Promotion Profit Agent",
        "script": PROJECT_ROOT
        / "agents"
        / "promotion_profit_agent.py",
    },
    {
        "name": "Basket Recommendation Agent",
        "script": PROJECT_ROOT
        / "agents"
        / "basket_recommendation_agent.py",
    },
    {
        "name": "Supervisor Agent",
        "script": PROJECT_ROOT
        / "agents"
        / "supervisor_agent.py",
    },
]


def run_step(step_number, step):
    name = step["name"]
    script = step["script"]

    print("\n" + "=" * 65)
    print(
        f"STEP {step_number}/{len(PIPELINE_STEPS)}: {name}"
    )
    print("=" * 65)

    if not script.exists():
        return {
            "name": name,
            "status": "FAILED",
            "duration_seconds": 0,
            "error": f"Script not found: {script}",
        }

    start_time = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    duration = round(
        time.perf_counter() - start_time,
        2,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)

        return {
            "name": name,
            "status": "FAILED",
            "duration_seconds": duration,
            "error": result.stderr[-3000:],
        }

    print(
        f"{name} completed in {duration:.2f} seconds"
    )

    return {
        "name": name,
        "status": "SUCCESS",
        "duration_seconds": duration,
        "error": None,
    }


def main():
    pipeline_start = time.perf_counter()

    print("=" * 65)
    print("RETAILGROWTH AI — MULTI-AGENT PIPELINE")
    print("=" * 65)

    results = []

    for number, step in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        result = run_step(number, step)
        results.append(result)

        if result["status"] == "FAILED":
            print(
                "\nPipeline stopped because a step failed."
            )
            break

    total_duration = round(
        time.perf_counter() - pipeline_start,
        2,
    )

    successful_steps = sum(
        result["status"] == "SUCCESS"
        for result in results
    )

    pipeline_status = (
        "SUCCESS"
        if successful_steps == len(PIPELINE_STEPS)
        else "FAILED"
    )

    run_log = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "pipeline_status": pipeline_status,
        "successful_steps": successful_steps,
        "total_steps": len(PIPELINE_STEPS),
        "duration_seconds": total_duration,
        "python_executable": sys.executable,
        "steps": results,
    }

    log_file = (
        REPORT_DIR / "pipeline_run_log.json"
    )

    with open(
        log_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_log,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Status: {pipeline_status}")
    print(
        f"Completed steps: "
        f"{successful_steps}/{len(PIPELINE_STEPS)}"
    )
    print(
        f"Total duration: {total_duration:.2f} seconds"
    )
    print(f"Run log: {log_file}")

    if pipeline_status == "SUCCESS":
        print(
            "\nAll agents completed successfully."
        )
        print(
            "Start the dashboard with: "
            "streamlit run app.py"
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
    