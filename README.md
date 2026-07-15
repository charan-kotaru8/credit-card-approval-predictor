[README.md](https://github.com/user-attachments/files/30051378/README.md)
# Credit Card Approval Prediction System

A machine learning–powered Flask web application that predicts whether a credit card application will be **Approved** or **Rejected**, trained on real-world applicant and credit history data. It automates the kind of screening banks do manually — evaluating income, employment, demographics, and payment history to flag high-risk applicants instantly instead of relying on slow, error-prone manual review.

Four classification algorithms are trained and compared — **Logistic Regression, Decision Tree, Random Forest, and XGBoost** — and the best performer (selected on minority-class F1, not raw accuracy, due to class imbalance) is served through the app. A scoring scaffold is included for deploying the same model to **IBM Watson Machine Learning**.

## Use Cases

- **Automated screening** — a credit analyst enters an applicant's profile and gets an instant Approve/Reject prediction to prioritize review.
- **High-risk / compliance review** — a compliance officer batch-screens applicants with past-due loan records; the pipeline converts multi-class payment status codes into a clear binary risk label.
- **Customer self-service** — a prospective customer checks their own eligibility before formally applying.

## Dataset

Real-world "Credit Card Approval Prediction" dataset, two source files:

| File | Rows | Description |
|---|---|---|
| `application_record.csv` | ~438K applicants | Demographics & financials: gender, income, income type, education, family status, housing type, employment, occupation |
| `credit_record.csv` | ~1M records | Monthly payment `STATUS` per applicant (days-past-due buckets, paid-off, or no active loan) |

**Target engineering:** an applicant is labeled high-risk (`TARGET=0`, Rejected) if they were ever 60+ days past due (`STATUS` in `{2,3,4,5}`); otherwise `TARGET=1` (Approved). After merging on applicant ID and de-duplicating, the labeled dataset has **~10,000 applicants**, with a realistic class imbalance of roughly **95–96% Approved vs. 4–5% Rejected**, which shapes how the models are trained and evaluated.

## Project Structure

```
credit-card-approval/
├── app.py                              # Flask web server — loads models/ artifacts,
│                                        # serves pages, scores one applicant, writes
│                                        # prediction history
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── data/
│   ├── datasets.zip                    # source bundle
│   └── datasets/
│       ├── application_record.csv
│       └── credit_record.csv
│
├── models/
│   ├── preprocessor.pkl                # fitted preprocessing pipeline (scaler + encoder)
│   ├── best_model.pkl                  # best trained classifier
│   ├── metadata.json                   # feature schema (categorical features/options, etc.)
│   ├── model_comparison.csv            # evaluation summary across all 4 models
│   └── prediction_history.json         # stored history of recent predictions
│
├── src/
│   ├── eda.py                          # exploratory data analysis utilities
│   ├── preprocessing.py                # feature engineering & preprocessing pipeline
│   ├── train_models.py                 # trains/evaluates candidate models, saves artifacts
│   └── ibm_watson_scoring_scaffold.py  # IBM Watson scoring deployment scaffolding
│
├── notebooks/
│   └── figures/                        # saved PNG plots: EDA charts, ROC curves,
│                                        # confusion matrices, feature importances
│
├── templates/
│   ├── index.html                      # landing page
│   ├── predict.html                    # input form
│   ├── result.html                     # prediction results page
│   └── history.html                    # prediction history page
│
└── static/
    ├── css/
    │   └── styles.css                  # styling
    └── js/                             # client-side scripts (if any)
```

## Machine Learning Pipeline

1. **Data understanding** — inspect shape, dtypes, missing values, duplicates in both source files.
2. **EDA** — count plots (gender, income type, education, family status, housing), distribution plots (income, age, employment years), target balance, correlation heatmap.
3. **Preprocessing & feature engineering**
   - Missing `OCCUPATION_TYPE` (~30%) filled with `'Unknown'` instead of dropping rows
   - Duplicate applicant rows removed
   - `DAYS_BIRTH` → `AGE_YEARS`, `DAYS_EMPLOYED` → `EMPLOYED_YEARS` (365243-day sentinel for pensioners/unemployed handled)
   - Binary Y/N and M/F fields encoded to 0/1
   - `ColumnTransformer`: `StandardScaler` on numeric features, `OneHotEncoder` on categoricals
   - Stratified 80/20 train/test split
4. **Model training** — Logistic Regression, Decision Tree, Random Forest, and XGBoost, all trained with class-balanced weighting to counteract the ~95/5 imbalance.
5. **Evaluation** — raw accuracy is misleading under this imbalance (predicting "Approved" for everyone would already score ~95%+), so the best model is selected using **F1-score on the minority (Rejected/high-risk) class** and ROC-AUC, alongside 5-fold stratified cross-validation, confusion matrices, and feature importance.
6. **Serving** — `app.py` loads `best_model.pkl` and `preprocessor.pkl` to make live predictions and appends each result to `prediction_history.json`.

## Web Application

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page / project overview (`index.html`) |
| `/predict` | GET, POST | Applicant input form (`predict.html`) and result (`result.html`) |
| `/history` | GET | Log of past predictions (`history.html`) |

The prediction form collects:
- **Numeric:** number of children, annual income, age, years employed, family member count
- **Binary (Yes/No):** owns car, owns realty, has work phone, has phone, has email
- **Categorical (dropdowns, sourced from `metadata.json`):** gender, income type, education level, family status, housing type, occupation type

## Getting Started

```bash
# Clone the repository
git clone <your-repo-url>
cd credit-card-approval

# Install dependencies
pip install -r requirements.txt

# (Re)build the pipeline from raw data, if needed
python src/eda.py
python src/preprocessing.py
python src/train_models.py

# Run the web app
python app.py
```

Then open `http://localhost:5000` in your browser.

## Deployment (IBM Watson Machine Learning)

`src/ibm_watson_scoring_scaffold.py` packages the trained model and preprocessing object and scores through the following pipeline:

```
Local development → Git push → GitHub repository → IBM Cloud →
IBM Watson Machine Learning Service → IBM Cloud Object Storage (model artifacts) →
Live web application
```

This allows the same model to serve real-time predictions from a hosted endpoint instead of (or in addition to) the local `best_model.pkl`.

## Testing

Manual and automated test cases cover the happy path, boundary numeric values, binary toggle behavior, categorical coverage (including the engineered `Unknown` occupation bucket), and input validation robustness. An automated `pytest` suite using Flask's `app.test_client()` is recommended to run these checks without manual UI interaction.

## Disclaimer

This is an educational/portfolio project. The dataset's engineered target is a proxy for creditworthiness based on payment history, not an actual bank approval label, and the model should not be used for real lending decisions.

## License

MIT
