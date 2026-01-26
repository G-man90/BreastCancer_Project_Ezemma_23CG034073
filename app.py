import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [
                float(request.form["radius"]),
                float(request.form["texture"]),
                float(request.form["perimeter"]),
                float(request.form["area"]),
                float(request.form["concavity"])
            ]

            values = np.array(values).reshape(1, -1)
            scaled = scaler.transform(values)
            result = model.predict(scaled)

            prediction = "Malignant" if int(result[0]) == 1 else "Benign"

        except Exception:
            prediction = "Invalid input. Please enter valid numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
