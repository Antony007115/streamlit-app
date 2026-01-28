from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Load data
data = pd.read_csv("business_data.csv")

# 2. Basic business calculations
data["revenue"] = data["selling_price"] * data["quantity_sold"]
data["profit"] = (data["selling_price"] - data["cost_price"]) * data["quantity_sold"]

# 3. Feature engineering (NEW FEATURES)
data["profit_margin"] = (data["selling_price"] - data["cost_price"]) / data["selling_price"]
data["sell_through_rate"] = data["quantity_sold"] / (data["quantity_sold"] + 10)
data["inactivity_ratio"] = data["days_inactive"] / (data["visit_count"] + 1)

total_profit = data["profit"].sum()

# 4. Features and target
feature_columns = [
    "days_inactive",
    "visit_count",
    "profit_margin",
    "sell_through_rate",
    "inactivity_ratio"
]

X = data[feature_columns]
y = data["churn"]

# 5. Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# 6. Predict churn
data["churn_prediction"] = model.predict(X)

# 7. Model accuracy
accuracy = accuracy_score(y, data["churn_prediction"])
print("Churn Model Accuracy:", accuracy)

# 8. Business suggestions
print("\n--- BUSINESS SUGGESTIONS ---")
for _, row in data.iterrows():
    if row["profit"] > 500:
        print(f"Increase stock for {row['item_name']}")

    if row["churn_prediction"] == 1:
        print(f"Churn risk detected for {row['item_name']} → offer discount or improve service")

# 9. Final output
print("\n--- FINAL DATA OUTPUT ---")
print(data[[
    "item_name",
    "revenue",
    "profit",
    "profit_margin",
    "sell_through_rate",
    "inactivity_ratio",
    "churn_prediction"
]])

print("\nTOTAL PROFIT:", total_profit)