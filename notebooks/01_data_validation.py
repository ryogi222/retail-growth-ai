from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "retail_transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "daily_product_sales.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

print("=" * 55)
print("RETAIL DATA VALIDATION")
print("=" * 55)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Transactions: {df['transaction_id'].nunique():,}")
print(f"Products: {df['product_id'].nunique()}")
print(f"Stores: {df['store_id'].nunique()}")
print(f"Customers: {df['customer_id'].nunique():,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Data-quality checks
missing_values = int(df.isna().sum().sum())
duplicate_lines = int(
    df.duplicated(
        subset=["transaction_id", "product_id"]
    ).sum()
)
invalid_quantity = int((df["quantity"] <= 0).sum())
invalid_price = int((df["selling_price"] <= 0).sum())
invalid_discount = int(
    ((df["discount_pct"] < 0) | (df["discount_pct"] > 1)).sum()
)

expected_revenue = (
    df["selling_price"] * df["quantity"]
).round(2)

expected_profit = (
    (df["selling_price"] - df["unit_cost"]) * df["quantity"]
).round(2)

incorrect_revenue = int(
    (~df["revenue"].round(2).eq(expected_revenue)).sum()
)

incorrect_profit = int(
    (~df["profit"].round(2).eq(expected_profit)).sum()
)

print("\nDATA QUALITY")
print(f"Missing values: {missing_values}")
print(f"Duplicate transaction-product lines: {duplicate_lines}")
print(f"Invalid quantities: {invalid_quantity}")
print(f"Invalid prices: {invalid_price}")
print(f"Invalid discounts: {invalid_discount}")
print(f"Incorrect revenue values: {incorrect_revenue}")
print(f"Incorrect profit values: {incorrect_profit}")

quality_issues = (
    missing_values
    + duplicate_lines
    + invalid_quantity
    + invalid_price
    + invalid_discount
    + incorrect_revenue
    + incorrect_profit
)

if quality_issues == 0:
    print("\nVALIDATION PASSED")
else:
    print(f"\nVALIDATION FOUND {quality_issues:,} ISSUES")

# Business KPIs
total_revenue = df["revenue"].sum()
total_profit = df["profit"].sum()
total_units = df["quantity"].sum()
average_basket = (
    df.groupby("transaction_id")["revenue"].sum().mean()
)
profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue > 0
    else 0
)

print("\nBUSINESS KPIs")
print(f"Total revenue: £{total_revenue:,.2f}")
print(f"Total profit: £{total_profit:,.2f}")
print(f"Total units sold: {total_units:,}")
print(f"Average basket value: £{average_basket:,.2f}")
print(f"Profit margin: {profit_margin:.2f}%")

# Aggregate transaction lines into daily product sales.
daily_sales = (
    df.groupby(
        [
            "date",
            "store_id",
            "product_id",
            "product_name",
            "category",
        ],
        as_index=False,
    )
    .agg(
        units_sold=("quantity", "sum"),
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        average_selling_price=("selling_price", "mean"),
        average_discount=("discount_pct", "mean"),
        promotion=("promotion", "max"),
        transaction_count=("transaction_id", "nunique"),
    )
    .sort_values(["store_id", "product_id", "date"])
)

daily_sales.to_csv(OUTPUT_FILE, index=False)

print("\nDAILY MODEL DATA")
print(f"Daily rows: {len(daily_sales):,}")
print(f"Saved to: {OUTPUT_FILE}")
print("\nPreview:")
print(daily_sales.head())