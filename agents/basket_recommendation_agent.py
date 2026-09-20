from pathlib import Path

import joblib
import pandas as pd
from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "retail_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading transaction data...")

df = pd.read_csv(DATA_FILE)

total_transactions = df["transaction_id"].nunique()

product_information = (
    df.groupby("product_name", as_index=False)
    .agg(
        product_id=("product_id", "first"),
        category=("category", "first"),
        unit_price=("unit_price", "first"),
        unit_cost=("unit_cost", "first"),
    )
    .set_index("product_name")
)

# Each row represents one basket.
basket = pd.crosstab(
    df["transaction_id"],
    df["product_name"],
).gt(0)

print(f"Transactions: {len(basket):,}")
print(f"Products: {len(basket.columns)}")
print("Finding frequent product combinations...")

frequent_itemsets = apriori(
    basket,
    min_support=0.015,
    use_colnames=True,
    max_len=3,
)

if frequent_itemsets.empty:
    raise ValueError(
        "No frequent product combinations were found."
    )

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.05,
)

# Keep simple one-product-to-one-product rules.
rules = rules[
    (rules["antecedents"].apply(len) == 1)
    & (rules["consequents"].apply(len) == 1)
    & (rules["confidence"] >= 0.15)
].copy()

if rules.empty:
    raise ValueError(
        "No association rules passed the selected thresholds."
    )

rules["source_product"] = rules["antecedents"].apply(
    lambda products: next(iter(products))
)

rules["recommended_product"] = rules["consequents"].apply(
    lambda products: next(iter(products))
)

# Add product price and cost.
rules["recommended_price"] = rules[
    "recommended_product"
].map(product_information["unit_price"])

rules["recommended_cost"] = rules[
    "recommended_product"
].map(product_information["unit_cost"])

rules["recommended_category"] = rules[
    "recommended_product"
].map(product_information["category"])

# Recommend a 10% cross-selling discount.
bundle_discount = 0.10

rules["bundle_price"] = (
    rules["recommended_price"]
    * (1 - bundle_discount)
).round(2)

rules["profit_per_add_on"] = (
    rules["bundle_price"]
    - rules["recommended_cost"]
).round(2)

rules["estimated_joint_transactions"] = (
    rules["support"] * total_transactions
).round().astype(int)

rules["estimated_bundle_profit"] = (
    rules["estimated_joint_transactions"]
    * rules["profit_per_add_on"]
).round(2)

# Score combines frequency, confidence, lift and profit.
rules["opportunity_score"] = (
    rules["support"]
    * rules["confidence"]
    * rules["lift"]
    * rules["profit_per_add_on"].clip(lower=0)
    * 1000
).round(2)

rules["recommendation"] = (
    "Customers buying "
    + rules["source_product"]
    + " should be offered "
    + rules["recommended_product"]
    + " with 10% off"
)

recommendations = rules[
    [
        "source_product",
        "recommended_product",
        "recommended_category",
        "support",
        "confidence",
        "lift",
        "recommended_price",
        "bundle_price",
        "profit_per_add_on",
        "estimated_joint_transactions",
        "estimated_bundle_profit",
        "opportunity_score",
        "recommendation",
    ]
].copy()

percentage_columns = ["support", "confidence"]

recommendations[percentage_columns] = (
    recommendations[percentage_columns] * 100
).round(2)

recommendations["lift"] = recommendations[
    "lift"
].round(2)

recommendations = recommendations.sort_values(
    ["opportunity_score", "lift"],
    ascending=False,
).reset_index(drop=True)

# Save all recommendations.
output_file = (
    REPORT_DIR / "basket_recommendations.csv"
)
recommendations.to_csv(output_file, index=False)

# Save the association-rule model.
model_bundle = {
    "frequent_itemsets": frequent_itemsets,
    "rules": rules,
    "basket_columns": list(basket.columns),
    "minimum_support": 0.015,
    "minimum_confidence": 0.15,
    "bundle_discount": bundle_discount,
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "basket_recommendation_model.joblib",
)

print("\nBASKET MODEL RESULTS")
print(
    f"Frequent itemsets: {len(frequent_itemsets):,}"
)
print(f"Association rules: {len(recommendations):,}")

print("\nTop basket opportunities:")
print(
    recommendations[
        [
            "source_product",
            "recommended_product",
            "support",
            "confidence",
            "lift",
            "profit_per_add_on",
            "opportunity_score",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(f"\nRecommendations saved to: {output_file}")
print(
    "Model saved to: "
    "models/basket_recommendation_model.joblib"
)
print("Basket Recommendation Agent completed successfully")
