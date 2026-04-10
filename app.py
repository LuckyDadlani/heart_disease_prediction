import os
import sys
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_cols = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']

def init_model():
    global model, scaler
    
    if not os.path.exists('framingham.csv'):
        print("ERROR: framingham.csv not found. Please ensure the dataset exists in the project root.", file=sys.stderr)
        return
        
    print("Loading data and training model...")
    try:
        df = pd.read_csv('framingham.csv')
        df = df.drop(columns=['education'], errors='ignore')
        df = df.rename(columns={'male': 'Sex_male'})
        df = df.dropna()

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
    except Exception as e:
        print(f"ERROR initializing model: {e}", file=sys.stderr)

# Initialize the model on startup
init_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Logistic Regression",
        "features": len(feature_cols)
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("index.html", error="Model is not initialized due to missing data or error.")
        
    try:
        age_input = float(request.form['age'])
        sex_input = float(request.form['sex'])
        cigs_input = float(request.form['cigsPerDay'])
        chol_input = float(request.form['totChol'])
        sysbp_input = float(request.form['sysBP'])
        glucose_input = float(request.form['glucose'])

        # Notice the order must match feature_cols
        patient_df = pd.DataFrame([[
            age_input, sex_input, cigs_input, chol_input, sysbp_input, glucose_input
        ]], columns=feature_cols)
        
        patient_scaled = scaler.transform(patient_df)
        
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
        return render_template("index.html", error=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False, port=5000)
