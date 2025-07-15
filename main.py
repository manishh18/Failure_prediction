from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import types

# Failure type labels (from model2)
label_mapping = {
    1: "Overstrain Failure",
    2: "Power Failure",
    3: "Random Failures",
    4: "Tool Wear",
    5: "Heat Dissipation Failure"
}

# Custom functions used inside preprocessing or model
def kelvin_to_celsius(k_temp):
    return k_temp - 273.15

def ordinal_encoding(X):
    mapping = {"L": 0, "M": 1, "H": 2}
    return X.replace(mapping)

# Register functions for joblib deserialization
module_name = "__main__"
if module_name not in sys.modules:
    sys.modules[module_name] = types.ModuleType(module_name)

setattr(sys.modules[module_name], "kelvin_to_celsius", kelvin_to_celsius)
setattr(sys.modules[module_name], "ordinal_encoding", ordinal_encoding)

# Load models and preprocessor
preprocessor = joblib.load("preprocessing.joblib")     # Preprocessing pipeline
model = joblib.load("model_failure.joblib")            # Binary model: Failure / No Failure
model2 = joblib.load("failure_type.joblib")            # Multiclass model: Type of failure

# Initialize FastAPI
app = FastAPI()

# Input schema
class InputData(BaseModel):
    air_temperature_K: float
    process_temperature_K: float
    rotational_speed_rpm: int
    torque_Nm: float
    tool_wear_min: int
    type: str  # "L", "M", or "H"

# Predict route
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Rename to match training column names
        df.rename(columns={
            "air_temperature_K": "Air temperature [K]",
            "process_temperature_K": "Process temperature [K]",
            "rotational_speed_rpm": "Rotational speed [rpm]",
            "torque_Nm": "Torque [Nm]",
            "tool_wear_min": "Tool wear [min]",
            "type": "Type"
        }, inplace=True)

        # Preprocess
        processed_input = preprocessor.transform(df)

        # Step 1: Binary classification (Failure / No Failure)
        failure_prediction = model.predict(processed_input)[0]

        if failure_prediction == 0:
            return {"prediction": "No Failure"}
        else:
            # Step 2: Multiclass classification (Type of failure)
            failure_type_prediction = model2.predict(processed_input)[0]
            failure_label = label_mapping.get(failure_type_prediction, "Unknown Failure Type")

            return {
                "prediction": "Failure Detected",
                "failure_type": failure_label
            }

    except Exception as e:
        return {"error": str(e)}
