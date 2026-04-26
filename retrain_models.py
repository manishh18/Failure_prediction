import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load processed data
df = pd.read_csv('data/data_processed.csv')

# Train binary model for failure detection
X_binary = df.drop(['Machine failure', 'failure_type'], axis=1)
y_binary = df['Machine failure']
model_failure = RandomForestClassifier(random_state=42)
model_failure.fit(X_binary, y_binary)
joblib.dump(model_failure, 'model_failure.joblib')

# Train multiclass model for failure type (only on failure cases)
failure_data = df[df['Machine failure'] == 1]
X_type = failure_data.drop(['Machine failure', 'failure_type'], axis=1)
y_type = failure_data['failure_type']
model_failure_type = RandomForestClassifier(random_state=42)
model_failure_type.fit(X_type, y_type)
joblib.dump(model_failure_type, 'failure_type.joblib')

print("Models retrained and saved.")