

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

st.markdown("# Manish's Plan to Stop Machine Breakdowns")
st.markdown("")
st.markdown("- **Here’s the Deal**: It’s 6:00 PM IST on Friday, June 13, 2025, and I’m Manish, the maintenance supervisor at a busy factory in India. Our machines keep breaking down, and it’s slowing us down big time. I’ve got a dataset of 10,000 machine runs with info on temperatures, speed, torque, tool wear, and failures. My job? Find the patterns to stop these breakdowns. Let’s dive in!")
st.markdown("---")
st.markdown("## My Quick Checks: Finding the Problem")
st.markdown("""# Manish's Plan to Stop Machine Breakdowns

- **Here’s the Deal**: It’s 6:00 PM IST on Friday, June 13, 2025, and I’m Manish, the maintenance supervisor at a busy factory in India. Our machines keep breaking down, and it’s slowing us down big time. I’ve got a dataset of 10,000 machine runs with info on temperatures, speed, torque, tool wear, and failures. My job? Find the patterns to stop these breakdowns. Let’s dive in!

---

## My Quick Checks: Finding the Problem

- **Check 1: How Often Do Machines Break?**  
  First, I need the basics. What’s the distribution of the `Machine failure` label in the dataset? How many machines failed, and how many didn’t? This will show if breakdowns are a rare issue or a big headache.

- **Check 2: What Products Are We Making?**  
 What is the distribution of the 'productID' variable in the dataset? How many instances are of low, medium, and high quality variants?

- **Check 3: How Are Machines Running? Any Weird Numbers?**  
  Let’s look at machine conditions. What’s the range of values for `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, and `Tool wear`? Are there any outliers in the dataset ?

- **Check 4: What Conditions Cause Failures?**  
  I need to spot red flags. Is there any correlation between the continuous variables and the `Machine failure` label? For example, does higher tool wear lead to more machine failures, or is something else the culprit?

- **Check 5: Does Product Type Change How Machines Run?**  
  Do product types affect machine conditions? Is there any correlation between the `Product ID` (`Type`) and the continuous variables? For example, is the `Rotational speed` higher for high-quality products than low-quality ones, or do some products stress the machines more?

- **Check 6: Are There Sneaky Patterns to Catch?**  
  Let’s look deeper. Are there any interactions or non-linear relationships between the variables that matter for predictive maintenance? For example, does the torque shoot up fast with rotational speed, or are there other patterns to help us predict breakdowns?

---
""")

# Load data
df = pd.read_csv('D:/PM/data.csv')

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
st.markdown(f"-The highest type of failure is HDF(Heat Dissipation Failure) with 1.15% failure rate.")

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

fig2 = px.pie(
    names=type_dist.index,
    values=type_dist.values,
    title='Product Type Distribution'
)
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

fig4 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=outlier_cols
)

for i, col in enumerate(outlier_cols):
    fig4.add_trace(
        go.Histogram(x=df[col], name=col),
        row=1,
        col=i+1
    )

fig4.update_layout(
    height=400,
    width=800,
    yaxis_title='Frequency',
    title="Histograms of Torque and Speed",
    showlegend=False
)
st.plotly_chart(fig4)


st.markdown("Rotational speed may or may not be actual outliers, therefore we'll keep them in the dataset for now. (same for torque )")

st.markdown("### ✅ Check 4: What Conditions Cause Failures?")
st.markdown("I need to spot red flags. Is there any correlation between the continuous variables and the `Machine failure` label? For example, does higher tool wear lead to more machine failures, or is something else the culprit?")

test_cols = num_cols

corr = df[num_cols + ['Machine failure']].corr()
fig5 = px.imshow(corr, text_auto=True, title="Correlation Heatmap", zmin=-1, zmax=1)
st.plotly_chart(fig5)

fig5.update_layout(
    title='Correlation Matrix',
    height=1200,
    width=1200
)



st.markdown("Let’s test the hypothesis that continuous variables influence failures using statistical tests.")
st.markdown("**Null Hypothesis**: There is no signifcant relationship between the different columns and Machine Failure. \n\n**Alternate Hypothesis**: There is a significant relationship between the different columns and the machine failure label.")

results = []
for col in test_cols:
    failed = df[df['Machine failure'] == 1][col]
    not_failed = df[df['Machine failure'] == 0][col]
    stat, p = ttest_ind(failed, not_failed)
    results.append((col, p))

for col, p in results:
    st.markdown(f"- **{col}**: p-value = {p:.4f} {'✅ Significant' if p < 0.05 else '❌ Not Significant'}")

st.markdown("**Conclusion**: test confirmed that `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, and  `Tool wear` have a strong link to machine failures.")
st.markdown("### ✅ Check 5: Does Product Type Change How Machines Run?")
st.markdown("  Do product types affect machine conditions? Is there any correlation between the `Product ID` (`Type`) and the continuous variables? For example, is the `Rotational speed` higher for high-quality products than low-quality ones, or do some products stress the machines more??")

num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

fig = make_subplots(
    rows=len(num_cols),
    cols=1,
    subplot_titles=num_cols,
    vertical_spacing=0.03
)

for i, col in enumerate(num_cols):
    fig.add_trace(
        go.Violin(
            x=df['Type'],
            y=df[col],
            name=col,
            box_visible=True,
            meanline_visible=True,
            showlegend=False
        ),
        row=i + 1,
        col=1
    )

fig.update_layout(
    height=2000,
    width=800,
    title_text="Distribution of Numeric Features by Product Type"
)

st.plotly_chart(fig)

st.markdown("**Conclusion**: low-quality products may stress machines more, and key conditions like tool wear significantly impact failures")
st.markdown("### ✅ Check 6: Are There Sneaky Patterns to Catch?")
st.markdown("  Let’s look deeper. Are there any interactions or non-linear relationships between the variables that matter for predictive maintenance? For example, does the torque shoot up fast with rotational speed, or are there other patterns to help us predict breakdowns?")

num_cols = df[['Air temperature [K]', 
               'Process temperature [K]', 
               'Rotational speed [rpm]', 
               'Torque [Nm]', 
               'Tool wear [min]']]

fig = sns.pairplot(num_cols)

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
