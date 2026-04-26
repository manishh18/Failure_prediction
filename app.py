import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
import requests
import matplotlib.pyplot as plt
import joblib

# Model loading and utilities
label_mapping = {
    1: "Overstrain Failure",
    2: "Power Failure",
    3: "Random Failures",
    4: "Tool Wear",
    5: "Heat Dissipation Failure"
}

def kelvin_to_celsius(k_temp):
    return k_temp - 273.15

def ordinal_encoding(X):
    mapping = {"L": 0, "M": 1, "H": 2}
    return X.replace(mapping)

# Load models
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessing.joblib")
    model = joblib.load("model_failure.joblib")
    model2 = joblib.load("failure_type.joblib")
    return preprocessor, model, model2

preprocessor, model_failure, model_failure_type = load_models()

# Set page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Sidebar for page navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Exploratory Data Analysis", "Model Evaluation Metrics", "Machine Failure Prediction"]
)

# Home Page
if page == "Home":
    st.title("🛠️ Predictive Maintenance Dashboard")
    st.markdown("##### Anticipating Failures Before They Happen")

    st.markdown("""
    This project explores machine learning approaches for **predicting industrial machine failures** using a synthetic dataset.  
    The goal is to:
    - **Predict if a machine will fail** (binary classification)
    - **Identify the cause of failure** (multiclass classification)

    This can help reduce downtime, improve safety, and optimize maintenance schedules.
    """)

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
        | **Process Temperature [K]** | Air temp + 10 K + small variation (σ = 1 K).                                                     |
        | **Rotational Speed [rpm]**| Generated from power with noise.                                                                   |
        | **Torque [Nm]**           | Normally distributed around 40 Nm (σ = 10), clipped to non-negative.                              |
        | **Tool Wear [min]**       | Indicates usage time; varies by quality level.                                                     |
        | **Machine Failure**       | Target label: 1 if any failure occurred, else 0.                                                   |
        """)

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

    st.info("🔍 Use the sidebar to explore data, train models, and view metrics.")

# EDA Page
elif page == "Exploratory Data Analysis":
    st.markdown("# Manish's Plan to Stop Machine Breakdowns")
    st.markdown("- **Here’s the Deal**: It’s 6:00 PM IST on Friday, June 13, 2025, and I’m Manish, the maintenance supervisor at a busy factory in India. Our machines keep breaking down, and it’s slowing us down big time. I’ve got a dataset of 10,000 machine runs with info on temperatures, speed, torque, tool wear, and failures. My job? Find the patterns to stop these breakdowns. Let’s dive in!")
    st.markdown("---")
    st.markdown("## My Quick Checks: Finding the Problem")
    st.markdown("""
    - **Check 1: How Often Do Machines Break?**  
      First, I need the basics. What’s the distribution of the `Machine failure` label in the dataset? How many machines failed, and how many didn’t? This will show if breakdowns are a rare issue or a big headache.

    - **Check 2: What Products Are We Making?**  
      What is the distribution of the 'productID' variable in the dataset? How many instances are of low, medium, and high quality variants?

    - **Check 3: How Are Machines Running? Any Weird Numbers?**  
      Let’s look at machine conditions. What’s the range of values for `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, and `Tool wear`? Are there any outliers in the dataset?

    - **Check 4: What Conditions Cause Failures?**  
      I need to spot red flags. Is there any correlation between the continuous variables and the `Machine failure` label? For example, does higher tool wear lead to more machine failures, or is something else the culprit?

    - **Check 5: Does Product Type Change How Machines Run?**  
      Do product types affect machine conditions? Is there any correlation between the `Product ID` (`Type`) and the continuous variables? For example, is the `Rotational speed` higher for high-quality products than low-quality ones, or do some products stress the machines more?

    - **Check 6: Are There Sneaky Patterns to Catch?**  
      Let’s look deeper. Are there any interactions or non-linear relationships between the variables that matter for predictive maintenance? For example, does the torque shoot up fast with rotational speed, or are there other patterns to help us predict breakdowns?
    """)
    st.markdown("---")

    # Load data (adjust path or use sample data; assuming data.csv is available)
    try:
        df = pd.read_csv('data.csv')  # Update with actual path if needed
    except FileNotFoundError:
        st.error("Dataset 'data.csv' not found. Please ensure the file is available in the working directory.")
        st.stop()

    # Create failure type
    def get_failure_type(row):
        if row['TWF'] == 1:
            return 'TWF'
        elif row['HDF'] == 1:
            return 'HDF'
        elif row['PWF'] == 1:
            return 'PWF'
        elif row['OSF'] == 1:
            return 'OSF'
        elif row['RNF'] == 1:
            return 'RNF'
        else:
            return 'no failure'

    df['failure_type'] = df.apply(get_failure_type, axis=1)
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    st.markdown("### ✅ Check 1: How Often Do Machines Break?")
    st.markdown("What’s the distribution of the `Machine failure` label in the dataset?")
    fail_rate = df['Machine failure'].mean() * 100
    success_rate = 100 - fail_rate
    fail_counts = df['failure_type'].value_counts(normalize=True) * 100

    st.markdown(f"- The success rate of the machine is **{success_rate:.2f}%**")
    st.markdown(f"- The highest type of failure is HDF (Heat Dissipation Failure) with 1.15% failure rate.")

    fig1 = px.histogram(df, x='failure_type', title='Failure Type Distribution')
    fig1.update_layout(xaxis_title='Failure Type', yaxis_title='Count')
    st.plotly_chart(fig1)

    st.markdown("### ✅ Check 2: What Products Are We Making?")
    st.markdown("Distribution of the 'productID' or `Type` variable:")
    type_dist = df['Type'].value_counts(normalize=True) * 100
    type_dist.index = type_dist.index.map({'L': 'Low', 'M': 'Medium', 'H': 'High'})

    st.markdown(f"- Low: **{type_dist['Low']:.1f}%**")
    st.markdown(f"- Medium: **{type_dist['Medium']:.1f}%**")
    st.markdown(f"- High: **{type_dist['High']:.1f}%**")

    fig2 = px.pie(names=type_dist.index, values=type_dist.values, title='Product Type Distribution')
    st.plotly_chart(fig2)

    st.markdown("### ✅ Check 3: How Are Machines Running? Any Weird Numbers?")
    num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    st.markdown("Let’s look at machine conditions. Are there any outliers?")

    fig3 = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols, vertical_spacing=0.04)
    for i, col in enumerate(num_cols):
        fig3.add_trace(go.Box(x=df[col], name=col), row=i+1, col=1)
    fig3.update_layout(height=1200, width=800, title="Box Plots of Continuous Variables")
    st.plotly_chart(fig3)

    outlier_cols = ["Torque [Nm]", "Rotational speed [rpm]"]
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=outlier_cols)
    for i, col in enumerate(outlier_cols):
        fig4.add_trace(go.Histogram(x=df[col], name=col), row=1, col=i+1)
    fig4.update_layout(height=400, width=800, yaxis_title='Frequency', title="Histograms of Torque and Speed", showlegend=False)
    st.plotly_chart(fig4)

    st.markdown("Rotational speed may or may not be actual outliers, therefore we'll keep them in the dataset for now. (same for torque)")

    st.markdown("### ✅ Check 4: What Conditions Cause Failures?")
    st.markdown("I need to spot red flags. Is there any correlation between the continuous variables and the `Machine failure` label? For example, does higher tool wear lead to more machine failures, or is something else the culprit?")

    corr = df[num_cols + ['Machine failure']].corr()
    fig5 = px.imshow(corr, text_auto=True, title="Correlation Heatmap", zmin=-1, zmax=1)
    fig5.update_layout(title='Correlation Matrix', height=600, width=600)
    st.plotly_chart(fig5)

    st.markdown("Let’s test the hypothesis that continuous variables influence failures using statistical tests.")
    st.markdown("**Null Hypothesis**: There is no significant relationship between the different columns and Machine Failure. \n\n**Alternate Hypothesis**: There is a significant relationship between the different columns and the machine failure label.")

    results = []
    for col in num_cols:
        failed = df[df['Machine failure'] == 1][col]
        not_failed = df[df['Machine failure'] == 0][col]
        stat, p = ttest_ind(failed, not_failed)
        results.append((col, p))

    for col, p in results:
        st.markdown(f"- **{col}**: p-value = {p:.4f} {'✅ Significant' if p < 0.05 else '❌ Not Significant'}")

    st.markdown("**Conclusion**: test confirmed that `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, and `Tool wear` have a strong link to machine failures.")

    st.markdown("### ✅ Check 5: Does Product Type Change How Machines Run?")
    st.markdown("Do product types affect machine conditions? Is there any correlation between the `Product ID` (`Type`) and the continuous variables? For example, is the `Rotational speed` higher for high-quality products than low-quality ones, or do some products stress the machines more?")

    fig = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols, vertical_spacing=0.03)
    for i, col in enumerate(num_cols):
        fig.add_trace(
            go.Violin(x=df['Type'], y=df[col], name=col, box_visible=True, meanline_visible=True, showlegend=False),
            row=i+1, col=1
        )
    fig.update_layout(height=2000, width=800, title_text="Distribution of NumericκάFeatures by Product Type")
    st.plotly_chart(fig)

    st.markdown("**Conclusion**: low-quality products may stress machines more, and key conditions like tool wear significantly impact failures")

    st.markdown("### ✅ Check 6: Are There Sneaky Patterns to Catch?")
    st.markdown("Let’s look deeper. Are there any interactions or non-linear relationships between the variables that matter for predictive maintenance? For example, does the torque shoot up fast with rotational speed, or are there other patterns to help us predict breakdowns?")

    num_cols_df = df[num_cols]
    fig = sns.pairplot(num_cols_df)
    st.pyplot(fig)

    st.markdown("Among all possible combinations of continuous variables, Rotational Speed vs Torque have a negative correlation and process temperature vs air temperature have a positive correlation.")

    st.markdown("## 🛠️ Final Conclusion")
    st.markdown("""
    Manish’s analysis reveals that:
    - Machine failures occur in **3.48%** of cases
    - Low-quality products cause higher stress on machines
    - Key features like **tool wear** and **torque** significantly influence failures

    He plans to monitor low-quality products closely and build a predictive system to minimize breakdowns and boost efficiency.
    """)

# Model Evaluation Metrics Page
elif page == "Model Evaluation Metrics":
    st.title("Model Evaluation Metrics")

    st.markdown("""
    We evaluated **four classification models** for both:
    - **Predicting Machine Failure** (Binary Classification)
    - **Classifying Type of Failure** (Multiclass Classification)

    The models tested:
    - Logistic Regression  
    - Support Vector Classifier (SVC)  
    - Decision Tree Classifier  
    - Random Forest Classifier  
    """)

    st.header("Machine Failure - Model Comparison")
    mf_metrics = pd.DataFrame({
        'Model': ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier'],
        'Accuracy': [0.838039, 0.960632, 0.990158, 0.992748],
        'Precision': [0.866273, 0.948553, 0.988803, 0.989851],
        'Recall': [0.894812, 0.994812, 0.996498, 0.999351],
        'F1 Score': [0.880311, 0.971132, 0.992636, 0.994579]
    })

    st.dataframe(
        mf_metrics.style.format({
            'Accuracy': "{:.3f}",
            'Precision': "{:.3f}",
            'Recall': "{:.3f}",
            'F1 Score': "{:.3f}"
        })
    )

    st.success("**Random Forest Classifier** gave the best performance for predicting machine failure.")

    st.subheader("Classification Report - Random Forest (Machine Failure)")
    mf_report = pd.DataFrame({
        "Precision": [0.998684, 0.989851, 0.994267, 0.992804],
        "Recall": [0.979602, 0.999351, 0.989477, 0.992748],
        "F1 Score": [0.989051, 0.994579, 0.991815, 0.992730],
        "Support": [3873, 7710, 11583, 11583]
    }, index=["Class 0 (No Failure)", "Class 1 (Failure)", "Macro Avg", "Weighted Avg"])

    st.dataframe(
        mf_report.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "Support": "{:,.0f}"
        })
    )

    st.header("Type of Failure - Model Comparison")
    tof_metrics = pd.DataFrame({
        'Model': ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier'],
        'Accuracy': [0.823016, 0.938962, 0.985151, 0.992057],
        'Precision': [0.815595, 0.940773, 0.985151, 0.992150],
        'Recall': [0.823535, 0.939600, 0.985267, 0.992177],
        'F1 Score': [0.817672, 0.936935, 0.985148, 0.992059]
    })

    st.dataframe(
        tof_metrics.style.format({
            'Accuracy': "{:.3f}",
            'Precision': "{:.3f}",
            'Recall': "{:.3f}",
            'F1 Score': "{:.3f}"
        })
    )

    st.success("**Random Forest Classifier** also performed best for identifying the type of failure.")

    st.subheader("Classification Report - Random Forest (Type of Failure)")
    tof_report = pd.DataFrame({
        "Precision": [0.996290, 0.994336, 0.994921, 0.991701, 0.975782, 0.998400, 0.991905, 0.991907],
        "Recall": [1.000000, 0.998965, 1.000000, 0.998433, 1.000000, 0.954128, 0.991921, 0.991798],
        "F1 Score": [0.998142, 0.996645, 0.997454, 0.995056, 0.987743, 0.975762, 0.991800, 0.991738],
        "Support": [1880, 1933, 1959, 1915, 1934, 1962, 11583, 11583]
    }, index=["0 (TWF)", "1 (HDF)", "2 (OSF)", "3 (RNF)", "4 (PWF)", "5 (No Failure)", "Macro Avg", "Weighted Avg"])

    st.dataframe(
        tof_report.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "Support": "{:,.0f}"
        })
    )

# Machine Failure Prediction Page
elif page == "Machine Failure Prediction":
    st.title("🛠️ Machine Failure Prediction")

    type_display_to_backend = {
        "Low": "L",
        "Medium": "M",
        "High": "H"
    }

    with st.form("prediction_form"):
        st.markdown("### 🧾 Enter Machine Sensor Readings:")
        air_temperature_K = st.number_input(
            "Air Temperature (K)", 
            value=298.7, min_value=100.0, max_value=320.0, step=0.1, format="%.2f"
        )
        process_temperature_K = st.number_input(
            "Process Temperature (K)", 
            value=305.1, min_value=100.0, max_value=340.0, step=0.1, format="%.2f"
        )
        rotational_speed_rpm = st.number_input(
            "Rotational Speed (rpm)", 
            value=1500.0, min_value=100.0, max_value=3000.0, step=1.0, format="%.0f"
        )
        torque_Nm = st.number_input(
            "Torque (Nm)", 
            value=40.3, min_value=1.0, max_value=80.0, step=0.1, format="%.2f"
        )
        tool_wear_min = st.number_input(
            "Tool Wear (minutes)", 
            value=150.0, min_value=0.0, max_value=250.0, step=1.0, format="%.0f"
        )
        type_display = st.selectbox(
            "Product Quality Type", 
            options=["Low", "Medium", "High"]
        )
        type_input = type_display_to_backend[type_display]

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        input_dict = {
            "air_temperature_K": air_temperature_K,
            "process_temperature_K": process_temperature_K,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_Nm": torque_Nm,
            "tool_wear_min": tool_wear_min,
            "type": type_input,
        }

        try:
            df = pd.DataFrame([input_dict])
            df.rename(columns={
                "air_temperature_K": "Air temperature [K]",
                "process_temperature_K": "Process temperature [K]",
                "rotational_speed_rpm": "Rotational speed [rpm]",
                "torque_Nm": "Torque [Nm]",
                "tool_wear_min": "Tool wear [min]",
                "type": "Type"
            }, inplace=True)
            processed_input = preprocessor.transform(df)
            failure_prediction = model_failure.predict(processed_input)[0]
            if failure_prediction == 0:
                prediction = "No Failure"
                failure_type = None
            else:
                failure_type_prediction = model_failure_type.predict(processed_input)[0]
                prediction = "Failure Detected"
                failure_type = label_mapping.get(failure_type_prediction, "Unknown Failure Type")

            st.success("Prediction Results:")
            st.write(f"**Prediction:** {prediction}")
            if failure_type:
                st.write(f"**Failure Type:** {failure_type}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")