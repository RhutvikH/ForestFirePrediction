from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_regressor = pickle.load(open("models/ridge.pkl", "rb"))
elasticnet_regressor = pickle.load(open("models/elasticnet.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predictfires", methods=["GET", "POST"])
def make_prediction():

    if request.method == "POST":
        temperature: float = float(request.form.get('Temperature'))
        rh: float = float(request.form.get('RH'))
        ws: float = float(request.form.get('Ws'))
        rain: float = float(request.form.get('Rain'))
        ffmc: float = float(request.form.get('FFMC'))
        dmc: float = float(request.form.get('DMC'))
        isi: float = float(request.form.get('ISI'))
        classes: float = float(request.form.get('Classes'))
        region: float = float(request.form.get('Region'))

        X: list = [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]

        X_scaled: np.ndarray = scaler.transform(X)
        # result: float = ridge_regressor.predict(X_scaled)
        result: float = elasticnet_regressor.predict(X_scaled)

        return render_template("predict-fires.html", result = np.maximum(result[0], 0))

    else:
        return render_template("predict-fires.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
