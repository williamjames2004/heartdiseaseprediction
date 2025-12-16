from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Get JSON data from frontend
        data = request.get_json()
        print("Received data:", data)

        # 2️⃣ Convert to DataFrame (IMPORTANT)
        df = pd.DataFrame([data])

        # 3️⃣ Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # 4️⃣ Prepare response
        result = {
            "prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "probability": round(probability * 100, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)