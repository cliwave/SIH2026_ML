from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained ML model
model = joblib.load("models/water_quality_model.joblib")


@app.route("/")
def home():
    return "Water Quality ML API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    pH = data["pH"]
    turbidity = data["turbidity"]
    tds = data["tds"]
    temperature = data["temperature"]

    input_data = pd.DataFrame([{
        "pH": pH,
        "turbidity": turbidity,
        "tds": tds,
        "temperature": temperature
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        status = "NORMAL"
    else:
        status = "UNUSUAL"

    return jsonify({
        "status": status,
        "message": (
            "Water quality pattern is normal."
            if status == "NORMAL"
            else "Unusual water quality pattern detected."
        ),
        "readings": {
            "pH": pH,
            "turbidity": turbidity,
            "tds": tds,
            "temperature": temperature
        }
    })


if __name__ == "__main__":
    app.run(debug=True)