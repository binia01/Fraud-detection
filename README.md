# Banking-Fraud-Detection-ML

This repository contains the code and documentation for a machine learning project focused on improving fraud detection models for e-commerce and bank credit card transactions. The project addresses critical challenges such as class imbalance and the trade-off between over-detection (false positives) and missed fraud (false negatives).

---
## Project Structure

```
.
├── README.md
├── data/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
│
├── notebooks/ 
│   ├── eda-fraud-data.ipynb
│   └── eda-creditcard.ipynb
└── src/
    ├── data_loader.py
    ├── utils.py
    └── preprocessor.py
```

---

## Overall Project Objective

The overarching goal is to develop accurate and robust fraud detection models that effectively handle the unique challenges of both e-commerce and bank transaction data. This includes leveraging geolocation analysis and transaction pattern recognition, while carefully balancing security needs with user experience by minimizing false positives and preventing financial losses from false negatives.

---

## Table of Contents

- [Task 1: Data Analysis and Preprocessing](#task-1-data-analysis-and-preprocessing)
- [Task 2: Model Building and Training](#task-2-model-building-and-training)
- [Task 3: Model Explainability](#task-3-model-explainability)
- [How to Set Up and Run the Code](#how-to-set-up-and-run-the-code)
- [Project Structure](#project-structure)

---

## Task 1: Data Analysis and Preprocessing

**Objective:**  
Prepare raw transaction data for machine learning model building. This includes ensuring data quality, enriching datasets with new insights, and transforming features into a format suitable for algorithms.

**Datasets Used:**
- `Fraud_Data.csv`: E-commerce transaction data.
- `IpAddress_to_Country.csv`: Mapping of IP address ranges to countries.
- `creditcard.csv`: Bank credit card transaction data.

**Steps Performed:**

1. **Handle Missing Values**
    - Robust handling for missing critical identifiers (`ip_address`, `device_id`), with imputation for `sex` (mode) and `age` (median) as needed.
    - No missing values found in `IpAddress_to_Country.csv` and `creditcard.csv`.

2. **Data Cleaning**
    - Removed 1081 duplicates from `creditcard.csv`; no duplicates in other files.
    - Converted data types as appropriate, including datetime and integer conversions for accurate processing.

3. **Exploratory Data Analysis (EDA)**
    - Univariate and bivariate analysis to reveal dataset characteristics and patterns.
    - Severe class imbalance confirmed in both datasets.
    - Identification of dominant categories and key numerical/categorical distributions.

4. **Merge Datasets for Geolocation**
    - Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to enrich with country info.
    - Unmapped IPs (~14.5%) set to `'Unknown'` to prevent data loss.

5. **Feature Engineering**
    - E-commerce: Created `hour_of_day`, `day_of_week`, `time_since_signup`, and transaction frequency features.
    - Bank: Relied on rich PCA-transformed features already present.

6. **Data Transformation**
    - Stratified train-test split (70-30).
    - Imbalance handling: SMOTE for e-commerce, random undersampling for bank data.
    - Standard scaling of numerical features.
    - OneHotEncoding of categorical features for e-commerce data.

---

## Task 2: Model Building and Training

**Objective:**
Develop and evaluate machine learning models to detect fraudulent transactions. We experimented with multiple algorithms to find the best balance between precision and recall, given the highly imbalanced nature of the data.

**Models Trained:**
1.  **Logistic Regression:** A baseline linear model.
2.  **Random Forest:** An ensemble learning method for classification.
3.  **XGBoost:** A gradient boosting framework known for speed and performance.

**Performance Results:**

### 1. Credit Card Fraud Detection (Bank Data)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9737 | 0.0530 | 0.8737 | 0.1000 | 0.9620 |
| **Random Forest** | 0.9995 | 0.9103 | 0.7474 | 0.8208 | 0.9761 |
| **XGBoost** | 0.9991 | 0.6909 | 0.8000 | 0.7415 | 0.9723 |

*Key Takeaway:* Random Forest achieved the highest F1 Score (0.82) and Precision, making it a strong candidate for minimizing false positives while maintaining good detection rates. Logistic Regression had high recall but very low precision, leading to many false alarms.

### 2. E-commerce Fraud Detection

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7212 | 0.1987 | 0.6519 | 0.3046 | 0.7363 |
| **Random Forest** | 0.9554 | 0.9927 | 0.5272 | 0.6887 | 0.7682 |
| **XGBoost** | 0.9522 | 0.9282 | 0.5300 | 0.6748 | 0.7661 |

*Key Takeaway:* The E-commerce dataset was more challenging. Random Forest and XGBoost performed similarly, with very high precision but moderate recall (~53%). This suggests that while the models are very confident when they flag a fraud, they might be missing about half of the actual fraud cases. Future work could focus on improving recall for this dataset.

---

## How to Set Up and Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Fraud-Detection.git
    cd Fraud-Detection
    ```

2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Place Data Files:**
    - Ensure your `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` files are placed in the `/data/` directory (or update file paths in the scripts).
    - If you have intermediate files, include `Fraud_Data_merged.csv` and `creditcard_cleaned.csv` as well.

5. **Run the Scripts:**
    - The project is structured into logical steps. Run the Python scripts corresponding to each task:
        ```bash
        python script_name.py
        ```
    - If running in a notebook environment (e.g., Jupyter/Colab), execute the code cells sequentially.

---