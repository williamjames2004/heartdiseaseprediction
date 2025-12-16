import pandas as pd
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ================================
# Load dataset
# ================================
df = pd.read_csv("heart_disease.csv")

X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# ================================
# Define features correctly
# ================================
categorical_features = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

numerical_features = [
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total"
]

# ================================
# Preprocessing
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ================================
# Pipeline
# ================================
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", GaussianNB())
])

# ================================
# Train-test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)

# ================================
# Train
# ================================
model_pipeline.fit(X_train, y_train)

# ================================
# Evaluate
# ================================
y_pred = model_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Precision:", precision_score(y_test, y_pred) * 100)
print("Recall:", recall_score(y_test, y_pred) * 100)
print("F1 Score:", f1_score(y_test, y_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================================
# Test with ONE correct raw input
# ================================
new_patient = {
    "Age": 48,
    "Gender": "Male",
    "Weight": 78,
    "Height": 157,
    "BMI": 26.4,
    "Smoking": "Never",
    "Alcohol_Intake": "None",
    "Physical_Activity": "Sedentary",
    "Diet": "Healthy",
    "Stress_Level": "Medium",
    "Hypertension": 0,
    "Diabetes": 0,
    "Hyperlipidemia": 1,
    "Family_History": 1,
    "Previous_Heart_Attack": 0,
    "Systolic_BP": 104,
    "Diastolic_BP": 99,
    "Heart_Rate": 71,
    "Blood_Sugar_Fasting": 165,
    "Cholesterol_Total": 200
}

new_df = pd.DataFrame([new_patient])

prediction = model_pipeline.predict(new_df)[0]
probability = model_pipeline.predict_proba(new_df)[0][1]

print("\nPrediction:", "Heart Disease" if prediction == 1 else "Normal")
print("Risk Probability:", round(probability * 100, 2), "%")

# ================================
# Save model
# ================================
pickle.dump(model_pipeline, open("heart_model.pkl", "wb"))
print("\nModel saved as heart_model.pkl")