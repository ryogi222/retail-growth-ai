from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "retail_transactions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

# Monthly performance
monthly = (
    df.assign(month=df["date"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        units=("quantity", "sum"),
    )
)

# Product performance
products = (
    df.groupby(
        ["product_id", "product_name", "category"],
        as_index=False,
    )
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        units=("quantity", "sum"),
    )
    .sort_values("revenue", ascending=False)
)

# Store performance
stores = (
    df.groupby("store_id", as_index=False)
    .agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        transactions=("transaction_id", "nunique"),
    )
)

# Promotion comparison
promotions = (
    df.groupby("promotion", as_index=False)
    .agg(
        average_quantity=("quantity", "mean"),
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        rows=("transaction_id", "count"),
    )
)

promotions["promotion_type"] = promotions["promotion"].map(
    {0: "No Promotion", 1: "Promotion"}
)

# Save summary tables
monthly.to_csv(REPORT_DIR / "monthly_performance.csv", index=False)
products.to_csv(REPORT_DIR / "product_performance.csv", index=False)
stores.to_csv(REPORT_DIR / "store_performance.csv", index=False)
promotions.to_csv(
    REPORT_DIR / "promotion_performance.csv",
    index=False,
)

# Create dashboard
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Monthly revenue and profit
axes[0, 0].plot(
    monthly["month"],
    monthly["revenue"],
    marker="o",
    label="Revenue",
    color="#00539F",
)
axes[0, 0].plot(
    monthly["month"],
    monthly["profit"],
    marker="o",
    label="Profit",
    color="#E31837",
)
axes[0, 0].set_title("Monthly Revenue and Profit")
axes[0, 0].tick_params(axis="x", rotation=60)
axes[0, 0].legend()

# Top products
top_products = products.head(10).sort_values("revenue")
axes[0, 1].barh(
    top_products["product_name"],
    top_products["revenue"],
    color="#00539F",
)
axes[0, 1].set_title("Top 10 Products by Revenue")
axes[0, 1].set_xlabel("Revenue (£)")

# Store performance
x_positions = range(len(stores))

axes[1, 0].bar(
    [x - 0.2 for x in x_positions],
    stores["revenue"],
    width=0.4,
    label="Revenue",
    color="#00539F",
)
axes[1, 0].bar(
    [x + 0.2 for x in x_positions],
    stores["profit"],
    width=0.4,
    label="Profit",
    color="#E31837",
)
axes[1, 0].set_xticks(list(x_positions))
axes[1, 0].set_xticklabels(stores["store_id"], rotation=15)
axes[1, 0].set_title("Store Revenue and Profit")
axes[1, 0].legend()

# Promotion impact
axes[1, 1].bar(
    promotions["promotion_type"],
    promotions["average_quantity"],
    color=["#808080", "#E31837"],
)
axes[1, 1].set_title("Average Quantity: Promotion Comparison")
axes[1, 1].set_ylabel("Average quantity per product line")

fig.suptitle(
    "RetailGrowth AI – Business Performance Dashboard",
    fontsize=18,
    fontweight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_file = FIGURE_DIR / "retail_eda_dashboard.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close()

promotion_average = promotions.loc[
    promotions["promotion"] == 1,
    "average_quantity",
].iloc[0]

regular_average = promotions.loc[
    promotions["promotion"] == 0,
    "average_quantity",
].iloc[0]

promotion_uplift = (
    (promotion_average - regular_average)
    / regular_average
    * 100
)

print("Exploratory analysis completed")
print(f"Dashboard saved to: {output_file}")
print(f"Promotion quantity uplift: {promotion_uplift:.2f}%")
print("\nTop five products:")
print(
    products[
        ["product_name", "revenue", "profit", "units"]
    ].head()
)