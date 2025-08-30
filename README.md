# Transaction Classifier (Portfolio Project)

This repository contains an end-to-end transaction classification pipeline with:
- Synthetic data generation
- Preprocessing & feature engineering
- Training script with **sentence-transformers** support (if installed) or TF-IDF fallback
- Model artifact saved to `models/transaction_model.joblib`
- FastAPI inference app: `app/api.py`
- Streamlit demo UI: `app/ui_streamlit.py`
- Dockerfile, tests, and CI hints

## Quickstart (local)

1. Create virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Generate synthetic data (already generated in `data/transactions.csv`):
   ```bash
   python -c "from training.train_model import load_data; print('data ready')"
   ```

3. Train model (this trains a TF-IDF + RandomForest fallback if Sentence-Transformers is not available):
   ```bash
   python training/train_model.py
   ```

4. Run API:
   ```bash
   uvicorn app.api:app --reload
   ```

5. Run UI:
   ```bash
   streamlit run app/ui_streamlit.py
   ```

Notes:
- If you want to use **sentence-transformers** embeddings, install `sentence-transformers` and re-run `training/train_model.py`. The training script will detect and use SBERT automatically.
- The saved model is at `models/transaction_model.joblib` after training.
