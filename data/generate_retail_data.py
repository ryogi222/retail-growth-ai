from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_TRANSACTIONS = 25_000

rng = np.random.default_rng(RANDOM_SEED)

products = pd.DataFrame(
    [
        ["P001", "Whole Milk", "Dairy", 1.65, 1.05, 10],
        ["P002", "Cheddar Cheese", "Dairy", 3.25, 2.10, 5],
        ["P003", "Greek Yogurt", "Dairy", 1.80, 1.10, 6],
        ["P004", "White Bread", "Bakery", 1.40, 0.75, 9],
        ["P005", "Croissants", "Bakery", 2.20, 1.25, 4],
        ["P006", "Pasta", "Grocery", 1.20, 0.60, 7],
        ["P007", "Pasta Sauce", "Grocery", 1.85, 0.95, 6],
        ["P008", "Breakfast Cereal", "Grocery", 3.50, 2.00, 5],
        ["P009", "Basmati Rice", "Grocery", 4.50, 2.80, 5],
        ["P010", "Baked Beans", "Grocery", 1.10, 0.55, 7],
        ["P011", "Chicken Breast", "Fresh Food", 5.50, 3.75, 5],
        ["P012", "Apples", "Produce", 2.00, 1.10, 6],
        ["P013", "Bananas", "Produce", 1.30, 0.65, 8],
        ["P014", "Potatoes", "Produce", 2.50, 1.30, 6],
        ["P015", "Frozen Pizza", "Frozen", 4.25, 2.60, 5],
        ["P016", "Ice Cream", "Frozen", 3.75, 2.20, 4],
        ["P017", "Crisps", "Snacks", 2.00, 0.95, 7],
        ["P018", "Chocolate", "Snacks", 1.50, 0.70, 7],
        ["P019", "Orange Juice", "Drinks", 2.30, 1.25, 5],
        ["P020", "Soft Drink", "Drinks", 2.10, 1.00, 7],
    ],
    columns=[
        "product_id",
        "product_name",
        "category",
        "unit_price",
        "unit_cost",
        "popularity",
    ],
)

stores = ["Demo_Store_001", "Demo_Store_002", "Demo_Store_003"]

start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2026-08-31")
number_of_days = (end_date - start_date).days + 1

product_lookup = products.set_index("product_id")
product_ids = products["product_id"].to_numpy()
product_probabilities = (
    products["popularity"] / products["popularity"].sum()
).to_numpy()

linked_products = {
    "P004": "P001",  # Bread and milk
    "P006": "P007",  # Pasta and pasta sauce
    "P008": "P001",  # Cereal and milk
    "P015": "P020",  # Pizza and soft drink
    "P017": "P020",  # Crisps and soft drink
}

rows = []

for transaction_number in range(1, NUMBER_OF_TRANSACTIONS + 1):
    transaction_id = f"T{transaction_number:06d}"
    transaction_date = start_date + pd.Timedelta(
        days=int(rng.integers(0, number_of_days))
    )

    store_id = rng.choice(stores)
    customer_id = f"C{int(rng.integers(1, 5001)):05d}"

    basket_size = int(rng.integers(1, 6))
    basket = set(
        rng.choice(
            product_ids,
            size=basket_size,
            replace=False,
            p=product_probabilities,
        )
    )

    # Add realistic product relationships.
    for product_id, linked_product in linked_products.items():
        if product_id in basket and rng.random() < 0.60:
            basket.add(linked_product)

    is_weekend = transaction_date.dayofweek >= 5
    is_holiday_period = transaction_date.month in [11, 12]

    for product_id in basket:
        product = product_lookup.loc[product_id]

        promotion = rng.random() < 0.20
        discount_pct = (
            float(rng.choice([0.10, 0.15, 0.20, 0.25]))
            if promotion
            else 0.0
        )

        quantity_lambda = 1.1
        quantity_lambda += 0.40 if is_weekend else 0
        quantity_lambda += 0.50 if promotion else 0
        quantity_lambda += 0.25 if is_holiday_period else 0

        quantity = max(1, int(rng.poisson(quantity_lambda)))

        selling_price = round(
            product["unit_price"] * (1 - discount_pct),
            2,
        )
        revenue = round(selling_price * quantity, 2)
        profit = round(
            (selling_price - product["unit_cost"]) * quantity,
            2,
        )

        rows.append(
            {
                "transaction_id": transaction_id,
                "date": transaction_date.date(),
                "store_id": store_id,
                "customer_id": customer_id,
                "product_id": product_id,
                "product_name": product["product_name"],
                "category": product["category"],
                "quantity": quantity,
                "unit_price": product["unit_price"],
                "unit_cost": product["unit_cost"],
                "promotion": int(promotion),
                "discount_pct": discount_pct,
                "selling_price": selling_price,
                "revenue": revenue,
                "profit": profit,
                "is_weekend": int(is_weekend),
                "is_holiday_period": int(is_holiday_period),
            }
        )

transactions = pd.DataFrame(rows).sort_values(
    ["date", "transaction_id", "product_id"]
)

output_directory = Path(__file__).resolve().parent
output_file = output_directory / "retail_transactions.csv"

transactions.to_csv(output_file, index=False)

print("Dataset generated successfully")
print(f"File: {output_file}")
print(f"Rows: {len(transactions):,}")
print(f"Transactions: {transactions['transaction_id'].nunique():,}")
print(f"Revenue: £{transactions['revenue'].sum():,.2f}")
print(f"Profit: £{transactions['profit'].sum():,.2f}")
print("\nSample:")
print(transactions.head())