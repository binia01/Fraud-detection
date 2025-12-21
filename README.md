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
