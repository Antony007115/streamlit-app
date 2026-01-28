import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Smart Churn Profit Optimizer",
    layout="wide"
)

# Title
st.title("Smart Churn Profit Optimizer")
st.caption("Decision-support dashboard for small businesses")

st.divider()

# File upload
uploaded_file = st.file_uploader(
    "Upload business data (CSV format)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Required columns (including item_name)
    required_columns = {
        "item_name",
        "cost_price",
        "selling_price",
        "quantity_sold",
        "days_inactive",
        "visit_count",
        "churn"
    }

    if not required_columns.issubset(data.columns):
        st.error("Uploaded file does not contain all required columns.")
        st.stop()

    # Profit calculation
    data["profit"] = (
        data["selling_price"] - data["cost_price"]
    ) * data["quantity_sold"]

    total_profit = data["profit"].sum()

    # Churn model
    X = data[["days_inactive", "visit_count"]]
    y = data["churn"]

    model = LogisticRegression()
    model.fit(X, y)

    data["churn_prediction"] = model.predict(X)
    accuracy = accuracy_score(y, data["churn_prediction"])

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit", f"₹ {total_profit:.0f}")
    col2.metric("Model Accuracy (MVP)", f"{accuracy:.2f}")
    col3.metric("Total Records", len(data))

    st.divider()

    # Data preview
    st.subheader("Business Data Overview")
    st.dataframe(data, use_container_width=True)

    st.divider()

    # ---------------- PROFIT BY ITEM CHART ----------------
    st.subheader("Profit Contribution by Item")

    profit_by_item = (
        data.groupby("item_name")["profit"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(profit_by_item)

    st.divider()

    # High profit items
    st.subheader("High Focus Items")

    high_profit = data[data["profit"] > 500]

    if len(high_profit) > 0:
        for _, row in high_profit.iterrows():
            st.success(
                f"Increase stock for {row['item_name']} "
                f"(Profit: ₹{row['profit']})"
            )
    else:
        st.info("No high-profit items identified for the current data.")

    st.divider()

    # Churn risk section
    st.subheader("Churn Risk Customers")

    churn_risk = data[data["churn_prediction"] == 1]

    if len(churn_risk) > 0:
        for _ in churn_risk.iterrows():
            st.warning(
                "Customer shows churn risk based on recent activity. "
                "Consider retention actions such as offers or follow-ups."
            )
    else:
        st.success("No churn-risk customers detected.")

else:
    st.info("Upload a CSV file to start the analysis.")