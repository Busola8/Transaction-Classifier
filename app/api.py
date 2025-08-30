from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os, numpy as np, pandas as pd
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'transaction_model.joblib')
# load model if present
model_art = None
try:
    model_art = joblib.load(MODEL_PATH)
except Exception as e:
    print('Model not found at', MODEL_PATH, '— run training script to create it')

app = FastAPI(title='Transaction Classifier API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class Tx(BaseModel):
    tx_id: str = None
    timestamp: str
    amount: float
    merchant: str = ''
    description: str = ''
    channel: str = 'MOBILE'

def clean_text(s):
    import re
    if s is None: return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@app.get('/healthz')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(tx: Tx):
    if model_art is None:
        return {'error':'model not found. run training script first'}
    # prepare features
    merchant_desc = clean_text((tx.merchant or '') + ' ' + (tx.description or ''))
    hour = pd.to_datetime(tx.timestamp).hour if tx.timestamp else 0
    is_weekend = int(pd.to_datetime(tx.timestamp).dayofweek in [5,6]) if tx.timestamp else 0
    log_amount = np.log1p(tx.amount)
    # model might be dict with type
    if isinstance(model_art, dict):
        mtype = model_art.get('type')
        m = model_art.get('model')
        if mtype == 'sbert':
            # requires SBERT model in the training environment; not supported here
            return {'error':'SBERT model saved; please deploy with sentence-transformers installed'}
    else:
        m = model_art
    # create DataFrame for pipeline
    df = pd.DataFrame([{'merchant_desc': merchant_desc, 'log_amount': log_amount, 'channel': tx.channel, 'hour': hour, 'is_weekend': is_weekend}])
    pred = m.predict(df)[0]
    probs = None
    try:
        probs = m.predict_proba(df).tolist()[0]
    except Exception:
        probs = None
    return {'label': pred, 'probs': probs}
