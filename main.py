from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import types

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

module_name = "__main__"
if module_name not in sys.modules:
    sys.modules[module_name] = types.ModuleType(module_name)
setattr(sys.modules[module_name], "kelvin_to_celsius", kelvin_to_celsius)
setattr(sys.modules[module_name], "ordinal_encoding", ordinal_encoding)

preprocessor = joblib.load("preprocessing.joblib")
model = joblib.load("model_failure.joblib")
model2 = joblib.load("failure_type.joblib")

app = FastAPI()

class InputData(BaseModel):
    air_temperature_K: float
    process_temperature_K: float
    rotational_speed_rpm: int
    torque_Nm: float
    tool_wear_min: int
    type: str

@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.dict()
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
        failure_prediction = model.predict(processed_input)[0]
        if failure_prediction == 0:
            return {"prediction": "No Failure"}
        else:
            failure_type_prediction = model2.predict(processed_input)[0]
            failure_label = label_mapping.get(failure_type_prediction, "Unknown Failure Type")
            return {
                "prediction": "Failure Detected",
                "failure_type": failure_label
            }
    except Exception as e:
        return {"error": str(e)}