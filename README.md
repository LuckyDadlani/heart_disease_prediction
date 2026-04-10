# Heart Disease Prediction

## Description
This project implements a Logistic Regression machine learning model to predict the 10-year risk of developing Coronary Heart Disease (CHD). Built on the Framingham Heart Study dataset, the project includes both a comprehensive data pipeline script for training the model and evaluating its performance, as well as a Flask web application that serves a clinical risk assessment dashboard.

## Project Structure
```text
heart_disease_prediction/
├── .gitignore
├── app.py                  # Flask web application serving the model
├── framingham.csv          # Raw dataset from the Framingham Heart Study
├── heart_disease.py        # Main data processing and model training script
├── plots/                  # Directory containing generated EDA and evaluation plots
│   ├── plot1.png           # Target Class Distribution
│   ├── plot2.png           # Target Flag for Patients
│   ├── plot3.png           # Correlation Heatmap
│   ├── plot4.png           # Age Distribution by CHD Status
│   ├── plot5.png           # RFECV Accuracy vs Features
│   ├── plot6.png           # RFECV Feature Rankings
│   └── plot7.png           # Model Confusion Matrix
├── README.md               # Project documentation
├── requirements.txt        # Pinned Python dependencies
└── templates/
    └── index.html          # Web dashboard UI for the clinical assessment
```

## Features Used
The final model relies on 6 core clinical features automatically validated through feature selection:
- **Age**: Patient's age in years (1 to 120).
- **Sex (Sex_male)**: Biological sex of the patient (1 = Male, 0 = Female).
- **Cigarettes Per Day (cigsPerDay)**: Average number of cigarettes smoked per day.
- **Total Cholesterol (totChol)**: Total cholesterol level in mg/dL.
- **Systolic Blood Pressure (sysBP)**: Systolic blood pressure in mmHg.
- **Glucose**: Fasting blood glucose level in mg/dL.

## How to Install and Run

### 1. Set up the Environment
First, clone the repository and navigate to the project directory. Then, set up a virtual environment and install the pinned dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
Run the main script to load data, clean it, train the model, save visualizations, and open the interactive command-line prediction tool:

```bash
python heart_disease.py
```
*Note: This will generate all the plots and save them inside the `plots/` folder.*

### 3. Run the Web Dashboard
To start the Flask web application in production mode:

```bash
python app.py
```
Then, open your web browser and navigate to `http://127.0.0.1:5000/`. You can also verify the service status at `http://127.0.0.1:5000/health`.

## Model Performance
- **Algorithm**: Logistic Regression (max_iter=1000)
- **Accuracy**: ~84.90%
- **Note on Class Imbalance**: The Framingham dataset has a significant class imbalance (roughly 85% negative class / 15% positive class). Because of this, accuracy is artificially high. The model accurately predicts negative cases (high specificity) but struggles to catch all positive cases (low recall).

## Feature Selection
Selecting the right features is crucial to avoid noise in the model. In this project, we implemented two approaches:
1. **Manual Selection**: We manually selected 6 promising clinical indicators (`age`, `Sex_male`, `cigsPerDay`, `totChol`, `sysBP`, `glucose`) based on standard clinical cardiovascular risk factors.
2. **Recursive Feature Elimination (RFECV)**: We computationally verified feature importance using RFECV (with cross-validation). While RFECV suggested an optimal subset of 10 features for slightly higher cross-validated accuracy, we proceeded with our streamlined 6-feature model because it minimizes input friction for the user while retaining the most medically critical markers.

## Screenshots
Please see `plots/` folder for visualizations.

## Known Limitations
- **Imbalanced Dataset**: The lack of positive CHD cases in the dataset causes the model to prioritize negative predictions.
- **Static Dataset**: The model was trained on historical data from the Framingham Heart Study, which may not completely generalize to modern or highly diverse geographical populations.

## Future Improvements
- **Class Balancing (SMOTE)**: Implementing Synthetic Minority Over-sampling Technique (SMOTE) or adjusted class weights in Logistic Regression to improve the recall for high-risk patients.
- **Advanced Ensembles**: Switching to robust algorithms like Random Forest or XGBoost.
- **RFECV Incorporation**: The RFECV algorithm is already implemented in the script. In the future, the web app could be expanded to utilize the full 10-feature optimal set mathematically derived by RFECV.
- **Cloud Deployment**: Wrapping the Flask app in a Docker container and deploying it on a cloud platform (e.g., AWS, Render, or Heroku) for global accessibility.
