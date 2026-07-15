# TODO - Credit Card Approval Prediction System

- [ ] Create project scaffold under `credit-card-approval/` (data, models, notebooks/figures, src, templates, static, app.py).
- [ ] Create `requirements.txt` and `README.md`.
- [ ] Implement dataset download + merge + target engineering in `src/eda.py` and/or dedicated `src/data_utils.py`.
- [ ] Implement EDA plotting pipeline and save PNGs.
- [ ] Implement preprocessing (missing handling, feature engineering, ColumnTransformer, save `preprocessor.pkl` + `metadata.json`, processed CSV exports).
- [ ] Implement model training/evaluation for 4 models; save comparison tables, metrics plots, and best model as `best_model.pkl`.
- [ ] Implement Flask app (`app.py`) + templates + static CSS; wire to load artifacts and score single applicant.
- [ ] Add local prediction history (simple in-memory/file-based).
- [ ] Add IBM Watson ML deployment documentation + scaffolding script(s) for object storage upload and scoring endpoint usage.
- [ ] Add a one-command “train” path (e.g., `python -m src.train_models`) and verify `python app.py` works end-to-end (using existing artifacts or auto-train instructions).

