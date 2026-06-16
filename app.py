from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    cp = float(request.form['cp'])
    thalach = float(request.form['thalach'])
    oldpeak = float(request.form['oldpeak'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    age = float(request.form['age'])
    chol = float(request.form['chol'])

    features = np.array([[
        cp,
        thalach,
        oldpeak,
        ca,
        thal,
        age,
        chol
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = f"❤️ Heart Disease Detected ({confidence}% confidence)"
        color = "red"
    else:
        result = f"✅ No Heart Disease Detected ({confidence}% confidence)"
        color = "green"

    return render_template(
        "index.html",
        prediction_text=result,
        color=color,
        cp=cp,
        thalach=thalach,
        oldpeak=oldpeak,
        ca=ca,
        thal=thal,
        age=age,
        chol=chol
    )


if __name__ == '__main__':
    app.run(debug=True)