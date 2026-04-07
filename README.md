# Heart Disease Prediction using Logistic Regression

A Machine Learning project that predicts the 10-year risk of coronary heart disease (CHD) using Logistic Regression, based on the **Framingham Heart Study** dataset.

## Dataset Description

The dataset comes from the **Framingham Heart Study**, an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. It includes over **4,000 records** with **15 attributes** covering patient demographics, behavioral risk factors, and medical history.

**Target Variable:** `TenYearCHD` — Whether the patient developed coronary heart disease within 10 years (1 = Yes, 0 = No).

## Features Used for Prediction

The model uses the following **6 features** from the dataset:

| Feature      | Description                                      |
|-------------|--------------------------------------------------|
| `age`        | Age of the patient (in years)                    |
| `Sex_male`   | Gender (1 = Male, 0 = Female)                   |
| `cigsPerDay` | Number of cigarettes smoked per day              |
| `totChol`    | Total cholesterol level (mg/dL)                  |
| `sysBP`      | Systolic blood pressure (mmHg)                   |
| `glucose`    | Blood glucose level (mg/dL)                      |

## Model & Performance

- **Algorithm:** Logistic Regression (`max_iter=1000`)
- **Data Split:** 70% Training / 30% Testing (`random_state=4`)
- **Feature Scaling:** StandardScaler
- **Accuracy:** ~85% (exact value printed when script runs)

## Project Structure

```
heart_disease_prediction/
├── framingham.csv              # Original dataset (Framingham Heart Study)
├── heart_disease.py            # Main Python script with all ML code
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
├── .gitignore                  # Git ignore rules
├── plots/                      # Generated visualizations
│   ├── plot1_class_distribution.png
│   ├── plot2_chd_flag.png
│   ├── plot3_correlation_heatmap.png
│   ├── plot4_age_distribution.png
│   └── plot5_confusion_matrix.png
└── venv/                       # Python virtual environment (git-ignored)
```

## Plots Description

| Plot | Filename | Description |
|------|----------|-------------|
| 1 | `plot1_class_distribution.png` | Count plot showing class imbalance in TenYearCHD (how many patients have CHD vs don't) |
| 2 | `plot2_chd_flag.png` | Line plot of TenYearCHD flag across all patient indices |
| 3 | `plot3_correlation_heatmap.png` | Correlation heatmap of all features showing relationships between variables |
| 4 | `plot4_age_distribution.png` | Histogram of age distribution colored by CHD status |
| 5 | `plot5_confusion_matrix.png` | Confusion matrix heatmap showing model prediction performance |

## How to Set Up and Run

### 1. Clone or navigate to the project directory

```bash
cd heart_disease_prediction
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

### 5. Run the script

```bash
python heart_disease.py
```

The script will:
- Load and explore the dataset
- Clean the data
- Generate 5 plots (saved in the `plots/` folder)
- Train a Logistic Regression model
- Print accuracy, classification report, and confusion matrix
- Predict CHD risk for a sample patient

## Technologies Used

- **Python 3**
- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computing
- **scikit-learn** — Machine learning (Logistic Regression, StandardScaler, train/test split)
- **matplotlib** — Data visualization
- **seaborn** — Statistical data visualization
