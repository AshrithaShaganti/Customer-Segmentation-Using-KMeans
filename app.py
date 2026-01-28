import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🟢 Customer Segmentation Dashboard")

st.markdown(
    """
    **This system uses K-Means Clustering to group customers based on their purchasing
    behavior and similarities.**

    👉 Discover hidden customer groups without predefined labels.
    """
)

st.markdown("---")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale_customers_data.csv")

df = load_data()

numerical_features = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"
]

X = df[numerical_features]

# -----------------------------
# Input Section (MAIN PANEL)
# -----------------------------
st.subheader("🔧 Clustering Controls")

col1, col2, col3, col4 = st.columns(4)

with col1:
    feature_x = st.selectbox("Select Feature 1", numerical_features, index=0)

with col2:
    feature_y = st.selectbox("Select Feature 2", numerical_features, index=1)

with col3:
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)

with col4:
    random_state = st.number_input("Random State (Optional)", min_value=0, value=42)

st.markdown("<br>", unsafe_allow_html=True)

run_clustering = st.button("🟦 Run Clustering")

# -----------------------------
# Data Preparation
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Run Clustering
# -----------------------------
if run_clustering:

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.markdown("---")

    # -----------------------------
    # Visualization Section
    # -----------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots()  # 🔥 no fixed figsize

    ax.scatter(
        df[feature_x],
        df[feature_y],
        c=df["Cluster"],
        cmap="viridis"
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    fx_idx = numerical_features.index(feature_x)
    fy_idx = numerical_features.index(feature_y)

    ax.scatter(
        centers[:, fx_idx],
        centers[:, fy_idx],
        c="red",
        s=200,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Customer Segments")
    ax.legend()

    # 🔥 responsive plot
    st.pyplot(fig, use_container_width=True)

    # -----------------------------
    # Cluster Summary Section
    # -----------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_x, "mean"),
            Avg_Feature_2=(feature_y, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    st.subheader("💡 Business Interpretation")

    avg_ref = summary["Avg_Feature_1"].mean()

    for _, row in summary.iterrows():
        cid = int(row["Cluster"])

        if row["Avg_Feature_1"] > avg_ref:
            interpretation = "High-spending customers across selected categories"
            emoji = "🟢"
        elif row["Avg_Feature_1"] < avg_ref:
            interpretation = "Budget-conscious customers with lower spending"
            emoji = "🟡"
        else:
            interpretation = "Moderate spenders with selective purchasing behavior"
            emoji = "🔵"

        st.markdown(f"{emoji} **Cluster {cid}:** {interpretation}")

    # -----------------------------
    # User Guidance / Insight Box
    # -----------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning(
        "Select features, choose the number of clusters, and click **Run Clustering** to generate results."
    )
