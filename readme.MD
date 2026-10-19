# Telco Customer Churn — Classification Algorithms Comparison

A simple machine learning project that compares multiple classification algorithms using a real-world dataset and provides an interactive web interface.

## Dataset

**Kaggle: Telco Customer Churn (IBM / blastchar)**

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file and place it inside the `data/` directory with the following name:

```text
data/telco_churn.csv
```

## Project Structure

```text
churn_project/
├── data/
│   └── telco_churn.csv
├── models/                  # Best model and results table (generated after running train.py)
├── app.py                   # Streamlit web interface
├── dataprocessing.py        # Data cleaning and preprocessing
├── train.py                 # Model training and comparison
├── requirements.txt
└── readme.md
```

## Installation and Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train and compare all classification models:

```bash
python train.py
```

Launch the interactive web interface:

```bash
streamlit run app.py
```

## Classification Algorithms

The project compares the following algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Why F1-Score Instead of Accuracy?

The Telco Customer Churn dataset is imbalanced, with approximately 26% of customers belonging to the churn class.

Accuracy alone can be misleading when the classes are imbalanced. Therefore, the models are evaluated and compared using multiple metrics:

- **Precision:** Measures how many customers predicted to churn actually churned.
- **Recall:** Measures how many actual churn cases the model successfully identifies.
- **F1-Score:** Balances precision and recall, making it useful when both false positives and false negatives matter.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes across different classification thresholds.

The best-performing model is selected based on the chosen evaluation metric, with F1-score serving as the primary metric.