# Credit Card Approval Prediction System

End-to-end ML pipeline + Flask web app that predicts **Approved (1)** vs **Rejected (0/high-risk)** for credit card applicants.

## Local Setup

```bash
cd credit-card-approval
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the models (creates artifacts in `models/`)

```bash
python -m src.train_models
```

This will:
- download the dataset
- build binary target from `credit_record.csv`
- run EDA and save figures to `notebooks/figures/`
- fit preprocessing (`models/preprocessor.pkl`) and metadata (`models/metadata.json`)
- train/evaluate 4 models and save `models/best_model.pkl`

## Run the Flask app

```bash
python app.py
```

Open: http://127.0.0.1:5000

## Project Layout

- `data/` raw + processed CSVs
- `models/` preprocessor + best model + metadata + evaluation outputs
- `notebooks/figures/` saved plots
- `src/` python pipeline (EDA, preprocessing, training)
- `templates/` Flask HTML
- `static/` CSS/JS

