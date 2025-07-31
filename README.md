## Data Science Project Template
# CountReps 🏋️‍♂️  
*A Machine Learning-powered Repetition Counter for Barbell Exercises using Accelerometer and Gyroscope Data*

---

## 📌 Project Overview

**CountReps** is a data science project that leverages accelerometer and gyroscope sensor data to classify barbell exercises and count repetitions automatically. The project uses Python for data preprocessing, feature engineering, and model training, with final insights and results summarized in an interactive **Power BI dashboard**.

---

## 🎯 Goal

To build Python scripts that:
- Process raw accelerometer and gyroscope CSV data.
- Engineer relevant features for classification.
- Train a machine learning model to detect barbell exercise types and count reps.
- Visualize performance metrics and KPIs in Power BI.

---

## 📁 Folder Structure

```bash
CountReps/
│
├── data/
│   └── raw/
│       └── metamotion/      # Original raw sensor CSV files
│
├── docs/                    # Documentation files
├── models/                  # Trained ML models and checkpoints
├── notebooks/               # Jupyter notebooks for EDA and model experiments
├── references/              # Supporting materials or references
├── reports/
│   └── figures/             # Visual outputs and plots
├── src/                     # Core Python scripts
│   ├── data/                # Scripts for data loading and preprocessing
│   ├── features/            # Feature engineering and transformation
│   ├── models/              # Model training, evaluation, and prediction
│   └── visualization/       # Custom visualizations and plots
│
├── dash1.pbix               # Power BI dashboard summarizing project results
├── requirements.txt         # Python package dependencies
├── environment.yml          # Conda environment configuration
├── .gitignore
└── README.md                # Project overview (you're here!)

You can use this template to structure your Python data science projects. It is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

🧠 How It Works
1. Read Raw Data
Load CSV sensor data from data/raw/metamotion/.

2. Data Preprocessing
Clean and format timestamps, normalize signals, and merge multiple files into a unified DataFrame.

3. Feature Engineering
Extract features like signal magnitude, peak values, zero-crossings, etc., to feed into the ML model.

4. Model Training
Use supervised learning to train a classifier that identifies exercise types and counts reps.

5. Visualization & Reporting
All insights, model performance metrics, and exercise classification trends are displayed in the Power BI dashboard (dash1.pbix).

📊 Power BI Dashboard
The interactive dashboard dash1.pbix includes:

1. Total repetitions by exercise type

2. Model classification accuracy and confusion matrix

3. Time-series sensor signal visualization

4. Comparison of raw vs processed data

🛠️ Tech Stack
1. Languages & Tools: Python, Power BI

2. Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

3. Environment: Conda (environment.yml)

🚀 Getting Started

1. Clone the Repository
git clone https://github.com/RathoreRaghvendra/CountReps.git
cd CountReps

2. Set Up the Environment
Using Conda:
conda env create -f environment.yml
conda activate countreps

Or using pip:
pip install -r requirements.txt

3. Run the Pipeline
Use scripts from src/ to run data processing and modeling steps:
python src/data/load_data.py
python src/features/build_features.py
python src/models/train_model.py
