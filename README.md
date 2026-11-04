# Breast Cancer Classification – End-to-End Machine Learning Project

## 1. Project Overview

This project implements an end-to-end Machine Learning classification workflow using the Breast Cancer Wisconsin Diagnostic Dataset.

The main objective is to develop and evaluate multiple machine learning classification algorithms to predict whether a breast tumour is:

- **Benign (B):** Non-cancerous.
- **Malignant (M):** Cancerous.

Five machine learning classification algorithms are trained and evaluated using standard performance metrics.

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset

The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses.

Dataset information:

- Total instances: 569
- Total input features: 30
- Target variable: Diagnosis
- Classification type: Binary classification

### Target Variable Encoding

| Diagnosis | Encoded Value |
|---|---:|
| Benign (B) | 0 |
| Malignant (M) | 1 |

The ID column and unnecessary columns are removed during preprocessing.

## 3. Machine Learning Algorithms

The following five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler is applied to Logistic Regression, KNN, and SVM through Scikit-learn pipelines.

## 4. Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Remove unnecessary ID and empty columns.
3. Handle missing values.
4. Encode the target variable into numerical values.
5. Separate input features and target variable.
6. Convert input features into numerical format.
7. Split the dataset into training and testing sets.
8. Apply feature scaling where required.

### Train-Test Split

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratified sampling: Enabled

### Dataset Distribution

| Class | Number of Samples |
|---|---:|
| Benign (0) | 357 |
| Malignant (1) | 212 |
| **Total** | **569** |

## 5. Model Evaluation Metrics

Each trained model is evaluated using the following performance metrics:

- **Accuracy:** Measures the proportion of correctly classified samples.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures the proportion of predicted malignant cases that are actually malignant.
- **Recall:** Measures the proportion of actual malignant cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which measures the quality of binary classifications.

## 6. Project Structure


end-to-end-ml-classification/
│
├── model/
│   ├── common.py
│   ├── train_models.py
│   └── breast+cancer+wisconsin+diagnostic.csv
│
├──  model_artifacts/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── feature_columns.json
│   ├── class_names.json
│   └── metrics.json
│
├── test_data.csv
├── results_table.md
├── README.md
├── requirements.txt
└── .gitignore
```

## 7. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Matplotlib (if used for visualizations)
- Git
- GitHub
- Jupyter Notebook (if used for development)

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Manoj9977/end-to-end-ml-classification.git
```

### Step 2: Navigate to the Project Directory

```bash
cd end-to-end-ml-classification
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 9. Model Training

Execute the following command from the project root directory:

```bash
python model/train_models.py
```

This script performs the following operations:

1. Loads the Breast Cancer Wisconsin dataset.
2. Performs data preprocessing.
3. Splits the dataset into training and testing sets.
4. Trains all five classification algorithms.
5. Evaluates the trained models.
6. Saves the trained models in the `models/` directory.
7. Generates evaluation metrics and model comparison results.

## 10. Model Evaluation Results

All five classification models were evaluated using the same test dataset containing 114 samples.

The following table summarizes the actual model performance obtained during execution.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbours | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | 0.9756 | **0.9524** | **0.9639** | **0.9433** |

### Best Performing Model

Support Vector Machine achieved the highest test accuracy of **97.37%**.

Its other evaluation results are:

- AUC: 0.9947
- Precision: 0.9756
- Recall: 0.9524
- F1-score: 0.9639
- MCC: 0.9433

Logistic Regression achieved the highest AUC of 0.9960.

Based on the evaluation results, SVM demonstrated the strongest overall performance on this test split.

## 11. Generated Output Artifacts

The project generates the following output files:

| File | Description |
|---|---|
| logistic_regression.joblib | Trained Logistic Regression model |
| decision_tree.joblib | Trained Decision Tree model |
| knn.joblib | Trained KNN model |
| naive_bayes.joblib | Trained Gaussian Naive Bayes model |
| svm.joblib | Trained SVM model |
| feature_columns.json | Input feature information |
| class_names.json | Class label information |
| metrics.json | Evaluation metrics for all models |
| test_data.csv | Test dataset with target values |
| results_table.md | Model comparison table |

## 12. Conclusion

This project demonstrates an end-to-end machine learning classification workflow using the Breast Cancer Wisconsin Diagnostic Dataset.

Five classification algorithms were implemented, trained, and evaluated using multiple performance metrics.

The Support Vector Machine achieved the highest accuracy of 97.37% on the held-out test dataset, while Logistic Regression achieved the highest AUC of 0.9960.

The project demonstrates the practical implementation of data preprocessing, model training, model evaluation, and saving trained model artifacts using Python and Scikit-learn.



