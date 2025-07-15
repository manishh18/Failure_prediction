import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")


def main():
    st.title("Welcome to Home")

if __name__ == "__main__":
    main()

# -------------------- Title & Tagline --------------------
st.title("🛠️ Predictive Maintenance Dashboard")
st.markdown("##### Anticipating Failures Before They Happen")

# -------------------- Project Description --------------------
st.markdown("""
This project explores machine learning approaches for **predicting industrial machine failures** using a synthetic dataset.  
The goal is to:
- **Predict if a machine will fail** (binary classification)
- **Identify the cause of failure** (multiclass classification)

This can help reduce downtime, improve safety, and optimize maintenance schedules.
""")

# -------------------- Dataset Overview --------------------
st.subheader("📦 Dataset Overview")

with st.expander("Click to view dataset structure and features"):
    st.markdown("""
    The dataset simulates real-world manufacturing conditions with **10,000 data points** and **14 features**.  
    It includes sensor readings, operational settings, and failure indicators.

    #### 🔧 Features

    | **Feature**               | **Description**                                                                                     |
    |---------------------------|-----------------------------------------------------------------------------------------------------|
    | **UID**                   | Unique identifier (1 to 10,000).                                                                   |
    | **Product ID**            | Product quality variant (L: low, 50%; M: medium, 30%; H: high, 20%) with serial number.            |
    | **Air Temperature [K]**   | Random walk around 300 K (σ = 2 K).                                                                |
    | **Process Temperature [K]** | Air temp + 10 K + small variation (σ = 1 K).                                                       |
    | **Rotational Speed [rpm]**| Generated from power with noise.                                                                   |
    | **Torque [Nm]**           | Normally distributed around 40 Nm (σ = 10), clipped to non-negative.                              |
    | **Tool Wear [min]**       | Indicates usage time; varies by quality level.                                                     |
    | **Machine Failure**       | Target label: 1 if any failure occurred, else 0.                                                   |
    """)

# -------------------- Failure Modes --------------------
st.subheader("⚠️ Failure Modes")

with st.expander("Click to view failure conditions"):
    st.markdown("""
    A machine failure (label `1`) occurs if **any** of the following independent conditions are met:

    | **Failure Mode**          | **Condition**                                                                                     | **Occurrences** |
    |---------------------------|---------------------------------------------------------------------------------------------------|-----------------|
    | **Tool Wear Failure (TWF)** | Tool wear exceeds 200–240 mins.                                                                  | 120             |
    | **Heat Dissipation Failure (HDF)** | Air-process temp difference < 8.6 K and speed < 1380 rpm.                            | 115             |
    | **Power Failure (PWF)**   | Power (Torque × Speed in rad/s) < 3500 W or > 9000 W.                                             | 95              |
    | **Overstrain Failure (OSF)** | Wear × Torque > threshold (varies by quality level).                                         | 98              |
    | **Random Failure (RNF)**  | Random 0.1% probability.                                                                          | 5               |

    > **Note:** While the `Machine Failure` label is binary, the exact failure mode is also available as a multiclass label for root cause classification.
    """)

# -------------------- Navigation Tip --------------------
st.info("🔍 Use the sidebar to explore data, train models, and view metrics.")
