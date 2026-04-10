"""
Heart Disease Prediction Project
Author: Lucky Dadlani
Dataset: Framingham Heart Study
Model: Logistic Regression
"""

# ============================================================
# SECTION 1: Import Libraries
# ============================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFECV

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("   HEART DISEASE PREDICTION USING LOGISTIC REGRESSION")
print("   Based on the Framingham Heart Study Dataset")
print("=" * 60)

# ============================================================
# SECTION 2: Load the Dataset
# ============================================================
df = pd.read_csv('framingham.csv')

print("\n" + "=" * 60)
print("SECTION 2: DATA LOADING & EXPLORATION")
print("=" * 60)

print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

print(f"\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column Names ---")
print(list(df.columns))

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# ============================================================
# SECTION 3: Data Cleaning
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: DATA CLEANING")
print("=" * 60)

df = df.drop(columns=['education'])
print("\n✓ Dropped 'education' column")

df = df.rename(columns={'male': 'Sex_male'})
print("✓ Renamed 'male' column to 'Sex_male'")

rows_before = df.shape[0]
df = df.dropna()
rows_after = df.shape[0]
print(f"✓ Dropped rows with NaN values: {rows_before} → {rows_after} ({rows_before - rows_after} rows removed)")

print("\n--- TenYearCHD Value Counts (After Cleaning) ---")
print(df['TenYearCHD'].value_counts())

# ============================================================
# SECTION 4: Exploratory Data Analysis (EDA) — Plots
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Plot 1
plt.figure(figsize=(8, 5))
sns.countplot(x='TenYearCHD', data=df, palette='BuGn_r')
plt.title('Distribution of TenYearCHD (Class Imbalance)', fontsize=14)
plt.xlabel('TenYearCHD (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('plots/plot1.png', dpi=150)
plt.close()
print("✓ Plot 1 saved: plots/plot1.png")

# Plot 2
plt.figure(figsize=(12, 4))
plt.plot(df['TenYearCHD'].values, linewidth=0.5)
plt.title('TenYearCHD Flag for All Patient Indices', fontsize=14)
plt.xlabel('Patient Index', fontsize=12)
plt.ylabel('TenYearCHD', fontsize=12)
plt.tight_layout()
plt.savefig('plots/plot2.png', dpi=150)
plt.close()
print("✓ Plot 2 saved: plots/plot2.png")

# Plot 3
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of All Features', fontsize=14)
plt.tight_layout()
plt.savefig('plots/plot3.png', dpi=150)
plt.close()
print("✓ Plot 3 saved: plots/plot3.png")

# Plot 4
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='TenYearCHD', kde=True, palette='Set1')
plt.title('Age Distribution Colored by TenYearCHD', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('plots/plot4.png', dpi=150)
plt.close()
print("✓ Plot 4 saved: plots/plot4.png")

scaler = StandardScaler()

# ============================================================
# SECTION 5: RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 60)

all_feature_cols = ['Sex_male', 'age', 'cigsPerDay', 'currentSmoker', 'BPMeds',
                    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
                    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

X_all = df[all_feature_cols]
y_rfe = df['TenYearCHD']

X_all_scaled = scaler.fit_transform(X_all)
print("\n✓ All features scaled using StandardScaler")

rfecv_estimator = LogisticRegression(max_iter=1000)
rfecv_selector = RFECV(estimator=rfecv_estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv_selector.fit(X_all_scaled, y_rfe)

optimal_features_count = rfecv_selector.n_features_
print(f"\nOptimal number of features found by RFECV: {optimal_features_count}")

rfecv_selected_mask = rfecv_selector.support_
rfecv_rankings = rfecv_selector.ranking_
rfecv_selected_features = [col for col, selected in zip(all_feature_cols, rfecv_selected_mask) if selected]

print("\n--- RFECV Selected Features ---")
print(f"RFECV explicitly selected {optimal_features_count} feature(s):")
print(rfecv_selected_features)

# Comparison
print("\n--- Feature Selection Comparison ---")
feature_cols = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
manual_features = sorted(feature_cols)
rfecv_features_sorted = sorted(rfecv_selected_features)

print(f"  Manual selection chose ({len(feature_cols)}):   {feature_cols}")
print(f"  RFECV selected ({optimal_features_count}):         {rfecv_selected_features}")

if manual_features == rfecv_features_sorted:
    print("\n  ✅ Result: The manual and RFECV selections MATCH perfectly!")
    print("     Explanation: RFECV mathematically proved that our manual selection is optimal.")
else:
    print(f"\n  ⚠️  Result: The selections DIFFER.")
    print("     Explanation: RFECV found a different set of features yields better cross-validated accuracy than our manual list.")

# Plot 5: RFECV Accuracy vs Number of Features
plt.figure(figsize=(10, 6))
cv_scores = rfecv_selector.cv_results_['mean_test_score']
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', color='#3498db', linewidth=2)
plt.axvline(x=optimal_features_count, color='#e74c3c', linestyle=':', linewidth=2, label=f'Optimal ({optimal_features_count})')
plt.xlabel('Number of Features Selected', fontsize=12)
plt.ylabel('Cross-Validation Accuracy Score', fontsize=12)
plt.title('RFECV: Accuracy vs Number of Features', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('plots/plot5.png', dpi=150)
plt.close()
print("\n✓ Plot 5 saved: plots/plot5.png")

# Plot 6: RFECV Feature Rankings Bar Chart
plt.figure(figsize=(12, 6))
colors = ['#2ecc71' if s else '#e74c3c' for s in rfecv_selected_mask]
plt.barh(all_feature_cols, rfecv_rankings, color=colors)
plt.xlabel('RFECV Ranking (1 = Selected)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('RFECV Feature Rankings', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/plot6.png', dpi=150)
plt.close()
print("✓ Plot 6 saved: plots/plot6.png")

# ============================================================
# SECTION 6: Feature Selection & Data Preparation
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: FEATURE SELECTION & DATA PREPARATION")
print("=" * 60)

X = df[feature_cols]
y = df['TenYearCHD']

print(f"\nFeatures used: {feature_cols}")
print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

X_scaled = scaler.fit_transform(X)
print("\n✓ Features scaled using StandardScaler")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=4
)

print(f"\n--- Train/Test Split ---")
print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set:  X_test={X_test.shape}, y_test={y_test.shape}")

# ============================================================
# SECTION 7: Model Training
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: MODEL TRAINING")
print("=" * 60)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n✓ Logistic Regression model trained successfully (max_iter=1000)")

# ============================================================
# SECTION 8: Model Evaluation
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n*** Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%) ***")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No CHD (0)', 'Has CHD (1)']))

cm = confusion_matrix(y_test, y_pred)
print("--- Confusion Matrix ---")
print(cm)
print(f"\nInterpretation:")
print(f"  True Negatives  (TN) = {cm[0][0]:>5}  → Correctly predicted NO CHD")
print(f"  False Positives (FP) = {cm[0][1]:>5}  → Incorrectly predicted CHD (actually no CHD)")
print(f"  False Negatives (FN) = {cm[1][0]:>5}  → Incorrectly predicted NO CHD (actually has CHD)")
print(f"  True Positives  (TP) = {cm[1][1]:>5}  → Correctly predicted CHD")

# Plot 7: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No CHD (0)', 'Has CHD (1)'],
            yticklabels=['No CHD (0)', 'Has CHD (1)'])
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('plots/plot7.png', dpi=150)
plt.close()
print("\n✓ Plot 7 saved: plots/plot7.png")

# ============================================================
# SECTION 9: Custom Patient Prediction
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: CUSTOM PATIENT PREDICTION")
print("=" * 60)

patient_data = {
    'age': 55,
    'Sex_male': 1,
    'cigsPerDay': 20,
    'totChol': 250,
    'sysBP': 140,
    'glucose': 90
}

print("\n--- Patient Details ---")
for key, value in patient_data.items():
    print(f"  {key}: {value}")

patient_df = pd.DataFrame([patient_data])
patient_scaled = scaler.transform(patient_df[feature_cols])

prediction = model.predict(patient_scaled)[0]
probabilities = model.predict_proba(patient_scaled)[0]

print(f"\n--- Prediction Result ---")
if prediction == 1:
    print("⚠️  PREDICTION: This patient IS at risk of developing CHD in 10 years.")
else:
    print("✅ PREDICTION: This patient is NOT at risk of developing CHD in 10 years.")

print(f"\n--- Prediction Probabilities ---")
print(f"  Probability of No CHD (0): {probabilities[0]:.4f} ({probabilities[0] * 100:.2f}%)")
print(f"  Probability of CHD    (1): {probabilities[1]:.4f} ({probabilities[1] * 100:.2f}%)")

# ============================================================
# SECTION 10: Summary
# ============================================================
print("\n" + "=" * 60)
print("SECTION 10: PROJECT SUMMARY")
print("=" * 60)
print(f"\n  Dataset:        Framingham Heart Study")
print(f"  Records used:   {df.shape[0]}")
print(f"  Features:       {len(feature_cols)} ({', '.join(feature_cols)})")
print(f"  Model:          Logistic Regression")
print(f"  Accuracy:       {accuracy * 100:.2f}%")
print(f"  Plots saved:    7 (in plots/ folder)")
print("\n" + "=" * 60)
print("   PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

# ============================================================
# SECTION 11: Interactive Prediction
# ============================================================
print("\n" + "=" * 60)
print("INTERACTIVE PATIENT PREDICTION")
print("=" * 60)
print("Enter patient details when prompted.\n")

try:
    age_input = float(input("Enter patient's Age: "))
    sex_input = float(input("Enter Sex (1=Male, 0=Female): "))
    cigs_input = float(input("Enter Cigarettes Per Day (0 if non-smoker): "))
    chol_input = float(input("Enter Total Cholesterol (mg/dL): "))
    bp_input = float(input("Enter Systolic Blood Pressure (mmHg): "))
    glucose_input = float(input("Enter Glucose Level (mg/dL): "))

    interactive_patient = pd.DataFrame([[age_input, sex_input, cigs_input, chol_input, bp_input, glucose_input]], columns=feature_cols)
    interactive_scaled = scaler.transform(interactive_patient)
    interactive_result = model.predict(interactive_scaled)
    interactive_prob = model.predict_proba(interactive_scaled)

    print("\n--- PREDICTION RESULT ---")
    if interactive_result[0] == 1:
        print("PREDICTION: HIGH RISK — Patient may develop CHD in 10 years")
    else:
        print("PREDICTION: LOW RISK — Patient unlikely to develop CHD in 10 years")
    print(f"Probability of No CHD: {interactive_prob[0][0]*100:.2f}%")
    print(f"Probability of CHD:    {interactive_prob[0][1]*100:.2f}%")

except KeyboardInterrupt:
    print("\nInteractive prediction skipped.")
except ValueError as e:
    print(f"\nInvalid input: {e}")
except EOFError:
    print("\nInteractive prediction skipped (no input).")

print("\n" + "=" * 60)
print("PROJECT COMPLETE! All plots saved in the 'plots/' folder.")
print("=" * 60)