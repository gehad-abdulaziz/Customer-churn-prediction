
# Customer Churn Prediction

A machine learning project focused on predicting customer churn in banking services using advanced techniques.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## 🎯 Problem Statement

Customer churn is a critical challenge for financial institutions, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to develop a predictive model that can accurately identify customers at risk of churning, enabling proactive retention strategies.

## 📊 Dataset

The dataset contains 10,000 customer records with the following features:
- CreditScore: Customer's credit score
- Geography: Customer's country (France, Spain, Germany)
- Gender: Customer's gender (Male, Female)
- Age: Customer's age
- Tenure: Number of years the customer has been with the bank
- Balance: Account balance
- NumOfProducts: Number of products the customer has with the bank
- HasCrCard: Whether the customer has a credit card (1=Yes, 0=No)
- IsActiveMember: Whether the customer is an active member (1=Yes, 0=No)
- EstimatedSalary: Customer's estimated salary
- Exited: Whether the customer churned (1=Yes, 0=No) - Target variable

## 🔧 Methodology

### Data Preprocessing
- Removed irrelevant columns (RowNumber, CustomerId, Surname)
- Handled categorical variables using label encoding
- Applied feature scaling to numerical features

### Handling Imbalanced Data
- Identified significant class imbalance (20% churners vs 80% non-churners)
- Applied SMOTETomek technique to balance the dataset
- Resulted in 10,204 balanced samples for training

### Feature Engineering
Created 34 additional features including:
- BalancePerProduct: Balance divided by number of products
- TenureAgeRatio: Tenure divided by age
- CreditScore_Age: Product of credit score and age
- RiskScore: Composite risk indicator
- LoyaltyScore: Customer loyalty indicator
- EngagementScore: Customer engagement metric
- And more interaction and polynomial features

### Model Training
Trained and compared the following models:
- XGBoost (with hyperparameter optimization)
- LightGBM (with hyperparameter optimization)
- CatBoost
- Random Forest
- Stacking Ensemble (combining all above models)

### Evaluation Metrics
- Accuracy
- Precision
- Recall (primary focus for churn prediction)
- F1-Score
- ROC-AUC
- Confusion Matrix

## 📈 Results

The Stacking_Advanced model achieved the best performance for churn prediction:

| Model | Threshold | Accuracy | F1-Score | Precision | Recall | False Negatives | False Positives |
|--------|-----------|-----------|-----------|-----------|---------|-----------------|----------------|
| **Stacking_Advanced** | 0.28 | 0.8205 | 0.6136 | 0.5460 | **0.7002** | **122** | 237 |
| XGBoost_Optimized | 0.34 | 0.8295 | 0.6215 | 0.5668 | 0.6880 | 127 | 214 |
| WeightedEnsemble_Optimized | 0.34 | 0.8300 | 0.6205 | 0.5685 | 0.6830 | 129 | 211 |
| RandomForest | 0.40 | 0.8290 | 0.6140 | 0.5678 | 0.6683 | 135 | 207 |
| LightGBM_Optimized | 0.44 | 0.8500 | **0.6287** | **0.6334** | 0.6241 | 153 | **147** |
| CatBoost | 0.40 | 0.8420 | 0.6020 | 0.6176 | 0.5872 | 168 | 148 |

**Key Finding:** For churn prediction, recall is the most important metric as false negatives (missing actual churners) are more costly than false positives. The Stacking_Advanced model achieved the highest recall of 0.7002, successfully identifying 70% of customers who churned while missing only 122 out of 407 churners.

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Customer-churn-prediction.git
cd Customer-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📝 Usage

1. Place your dataset (churn.csv) in the project directory
2. Run the main script:
```bash
python churn_prediction.py
```

3. The script will:
   - Load and preprocess the data
   - Apply feature engineering
   - Handle class imbalance
   - Train and evaluate multiple models
   - Display performance metrics and visualizations

## 🧭 Future Work

- Implement real-time prediction API
- Incorporate additional customer behavioral data
- Develop a dashboard for monitoring churn risk
- Test model performance on more recent data
- Explore deep learning approaches

## 📚 References

- SMOTE: Synthetic Minority Over-sampling Technique
- XGBoost: Extreme Gradient Boosting
- LightGBM: Light Gradient Boosting Machine
- CatBoost: Categorical Boosting
- Stacking: Stacked Generalization

## 👤 Author

**Eng Gehad Abdulaziz**
- Data Scientist
- [GitHub Profile](https://github.com/gehad-abdulaziz)
- [LinkedIn Profile](https://www.linkedin.com/in/gehad-abdulaziz-228973287)

