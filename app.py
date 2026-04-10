import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def init_model():
    global model, scaler
    print("Loading data and training model...")
    df = pd.read_csv('framingham.csv')
    df = df.drop(columns=['education'])
    df = df.rename(columns={'male': 'Sex_male'})
    df = df.dropna()

    feature_cols = ['Sex_male', 'age', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']
    X = df[feature_cols]
    y = df['TenYearCHD']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=4
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model trained successfully!")

# Initialize the model on startup
init_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age_input = float(request.form['age'])
        sex_input = float(request.form['sex'])
        cigs_input = float(request.form['cigsPerDay'])
        stroke_input = float(request.form['prevalentStroke'])
        hyp_input = float(request.form['prevalentHyp'])
        chol_input = float(request.form['totChol'])
        sysbp_input = float(request.form['sysBP'])
        diabp_input = float(request.form['diaBP'])
        bmi_input = float(request.form['bmi'])
        glucose_input = float(request.form['glucose'])

        # Notice the order must match feature_cols
        patient_data = np.array([[sex_input, age_input, cigs_input, stroke_input, hyp_input, chol_input, sysbp_input, diabp_input, bmi_input, glucose_input]])
        patient_scaled = scaler.transform(patient_data)
        
        prediction = model.predict(patient_scaled)[0]
        probabilities = model.predict_proba(patient_scaled)[0]
        
        prob_0 = round(probabilities[0] * 100, 2)
        prob_1 = round(probabilities[1] * 100, 2)
        
        result_class = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        return render_template("index.html", 
                               result=result_class, 
                               prob_0=prob_0, 
                               prob_1=prob_1,
                               form_data=request.form)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
