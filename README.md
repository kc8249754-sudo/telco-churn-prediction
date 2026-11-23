# Telco Customer Churn Prediction: Model Evaluation

A machine-learning project that predicts whether a telecom customer will churn, comparing six classification algorithms with proper validation, overfitting/underfitting diagnosis, hyper-parameter tuning and a justified final model choice.

## Problem
Retaining an existing customer is cheaper than acquiring a new one. This project builds and compares models that flag customers likely to leave, so a company can target them with retention offers.

- **Task:** binary classification (`Churn`: Yes / No)
- **Data:** IBM Telco Customer Churn, 7,043 customers and 21 features (demographics, account details, services, charges)
- **Challenge:** class imbalance, with about 27% churners, so accuracy alone is misleading

## Approach
1. **Cleaning:** fixed `TotalCharges` (text with 11 blank values for new customers), dropped `customerID`, encoded the target.
2. **EDA:** churn rate by contract, internet service, payment method, tenure and charges.
3. **Leak-free pipeline:** scaling and one-hot encoding inside a scikit-learn `Pipeline`, so test data never influences preprocessing.
4. **Validation:** stratified 80/20 train/test split plus stratified 5-fold cross-validation.
5. **Models compared:** Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, Gradient Boosting.
6. **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices and ROC curves.
7. **Over/underfitting analysis:** train vs CV gap table, validation curve (tree depth) and learning curves.
8. **Tuning:** `GridSearchCV` on the strongest models, optimising ROC-AUC.
9. **Final evaluation:** a single run on the untouched test set, threshold analysis and permutation feature importance.

## Results

Cross-validated results on the training set (untuned models):

| Model | CV ROC-AUC | Train ROC-AUC | Gap | Diagnosis |
|---|---|---|---|---|
| Gradient Boosting | 0.848 | 0.888 | 0.04 | Good fit |
| Logistic Regression | 0.846 | 0.850 | 0.004 | Good fit |
| SVM (RBF) | 0.826 | 0.876 | 0.05 | Good fit |
| Random Forest | 0.824 | 1.000 | 0.18 | Overfitting |
| KNN | 0.824 | 0.868 | 0.04 | Good fit |
| Decision Tree | 0.661 | 1.000 | 0.34 | Overfitting |

Final test-set results (tuned models):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting (selected)** | 0.804 | 0.670 | 0.516 | 0.583 | 0.846 |
| Random Forest | 0.753 | 0.523 | 0.791 | 0.630 | 0.845 |
| Logistic Regression | 0.740 | 0.506 | 0.783 | 0.615 | 0.841 |
| Decision Tree (pruned) | 0.752 | 0.521 | 0.789 | 0.628 | 0.836 |
| SVM (RBF) | 0.746 | 0.514 | 0.775 | 0.618 | 0.822 |
| KNN | 0.779 | 0.586 | 0.564 | 0.575 | 0.818 |

### Key findings
- The unrestricted Decision Tree memorised the training data (train 1.00 vs CV 0.66). Limiting depth and leaf size reduced the train-CV gap from 0.34 to 0.03.
- **Selected model: tuned Gradient Boosting.** It had the highest cross-validated ROC-AUC (0.851), a small train-CV gap (0.026) and stable folds.
- The top three tuned models are within about 0.5% of each other, so Logistic Regression is a strong, more interpretable alternative.
- At the default 0.5 threshold the model misses almost half of churners. Lowering the threshold to 0.3 raises recall to about 0.77 (precision 0.54, F1 0.63), which suits a retention campaign.
- **Main churn drivers:** short tenure, month-to-month contracts, fiber-optic internet and electronic-check payment. Long contracts and tech support / online security are associated with lower churn.

## Repository contents
```
.
├── Telco_Churn_Model_Evaluation.ipynb   # full analysis with outputs
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # dataset (see Data section)
├── requirements.txt
└── README.md
```

## Data
IBM Telco Customer Churn dataset, available on Kaggle: `blastchar/telco-customer-churn`.
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` next to the notebook. If the file is missing, the notebook downloads the same IBM sample data from GitHub automatically.

## How to run

**Windows (PowerShell)**
```powershell
git clone https://github.com/kc8249754-sudo/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

**Mac / Linux**
```bash
git clone https://github.com/kc8249754-sudo/telco-churn-prediction.git
cd telco-churn-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `Telco_Churn_Model_Evaluation.ipynb` and choose **Run All**. Results may differ slightly between machines and library versions.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

## Limitations and future work
- Single data snapshot, with no time component and no business cost matrix.
- Planned: SMOTE inside cross-validation, XGBoost/LightGBM, SHAP explanations and cost-sensitive threshold selection.

## Author
Your Name, 3rd-year B.Tech CSE (AI/ML) student
[GitHub](https://github.com/kc8249754-sudo) · [LinkedIn](https://www.linkedin.com/in/your-profile)
