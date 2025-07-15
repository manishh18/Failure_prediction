import streamlit as st
import requests

st.title("🛠️ Machine Failure Prediction")

# Mapping shown label to backend-expected value
type_display_to_backend = {
    "Low": "L",
    "Medium": "M",
    "High": "H"
}

# Input form
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
    type_input = type_display_to_backend[type_display]  # send backend-compatible code

    submit = st.form_submit_button("🔍 Predict")

# On submission
if submit:
    payload = {
        "air_temperature_K": air_temperature_K,
        "process_temperature_K": process_temperature_K,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_Nm": torque_Nm,
        "tool_wear_min": tool_wear_min,
        "type": type_input,
    }

    try:
        # url = "http://127.0.0.1:8000/predict/"
        url = "http://localhost:8000/predict/"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction", "N/A")
        st.success(" Prediction Results:")
        st.write(f"**Prediction:** {prediction}")

        if prediction == "Failure Detected":
            failure_type = result.get("failure_type", "Unknown")
            st.write(f"**Failure Type:** {failure_type}")

    except requests.exceptions.RequestException as e:
        st.error(f" API request failed: {e}")
