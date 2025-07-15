import streamlit as st
import pandas as pd

st.title(" Model Evaluation Metrics")

# -------------------- Intro --------------------
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

# -------------------- Machine Failure Summary --------------------
st.header(" Machine Failure - Model Comparison")
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

st.success(" **Random Forest Classifier** gave the best performance for predicting machine failure.")

# -------------------- Machine Failure Classification Report --------------------
st.subheader(" Classification Report - Random Forest (Machine Failure)")

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

# -------------------- Type of Failure Summary --------------------
st.header(" Type of Failure - Model Comparison")
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

st.success(" **Random Forest Classifier** also performed best for identifying the type of failure.")

# -------------------- Type of Failure Classification Report --------------------
st.subheader(" Classification Report - Random Forest (Type of Failure)")

tof_report = pd.DataFrame({
    "Precision": [0.996290, 0.994336, 0.994921, 0.991701, 0.975782, 0.998400, 0.991905, 0.991907],
    "Recall": [1.000000, 0.998965, 1.000000, 0.998433, 1.000000, 0.954128, 0.991921, 0.991798],
    "F1 Score": [0.998142, 0.996645, 0.997454, 0.995056, 0.987743, 0.975762, 0.991800, 0.991738],
    "Support": [1880, 1933, 1959, 1915, 1934, 1962, 11583, 11583]
}, index=[
    "0 (TWF)", "1 (HDF)", "2 (OSF)", "3 (RNF)", "4 (PWF)", "5 (No Failure)", "Macro Avg", "Weighted Avg"
])

st.dataframe(
    tof_report.style.format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "Support": "{:,.0f}"
    })
)
