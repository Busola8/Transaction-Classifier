import os, re, joblib, json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'transaction_model.joblib')
VECT_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'text_vectorizer.joblib')

def clean_text(s):
    if pd.isna(s): return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_data():
    df = pd.read_csv(DATA_CSV)
    df['merchant_desc'] = (df['merchant'].fillna('') + ' ' + df['description'].fillna('')).apply(clean_text)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['log_amount'] = np.log1p(df['amount'].astype(float))
    X = df[['merchant_desc','log_amount','channel','hour','is_weekend']]
    y = df['label']
    return X, y, df

def try_sentence_transformer(X_train_text, X_test_text):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_train = model.encode(X_train_text.tolist(), show_progress_bar=True)
        emb_test = model.encode(X_test_text.tolist(), show_progress_bar=True)
        return emb_train, emb_test, True
    except Exception as e:
        print('SentenceTransformer not available or failed:', e)
        return None, None, False

def main():
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Try sentence-transformers
    emb_train, emb_test, ok = try_sentence_transformer(X_train['merchant_desc'], X_test['merchant_desc'])
    if ok:
        # concatenate numeric features
        import numpy as np
        num_cols = np.vstack([X_train[['log_amount','hour','is_weekend']].values, X_test[['log_amount','hour','is_weekend']].values])
        # but we'll split appropriately
        train_num = X_train[['log_amount','hour','is_weekend']].values
        test_num = X_test[['log_amount','hour','is_weekend']].values
        X_train_feat = np.hstack([emb_train, train_num])
        X_test_feat = np.hstack([emb_test, test_num])
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        clf.fit(X_train_feat, y_train)
        preds = clf.predict(X_test_feat)
        print(classification_report(y_test, preds))
        joblib.dump({'model':clf, 'type':'sbert'}, MODEL_OUT)
        print('Saved SBERT-based model to', MODEL_OUT)
    else:
        # Fallback TF-IDF pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        text_pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))])
        num_pipe = Pipeline([('scaler', StandardScaler())])
        pre = ColumnTransformer([
            ('text', text_pipe, 'merchant_desc'),
            ('num', num_pipe, ['log_amount']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['channel']),
            ('time', 'passthrough', ['hour','is_weekend'])
        ])
        clf = Pipeline([('pre', pre), ('model', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))
        joblib.dump({'model':clf, 'type':'tfidf'}, MODEL_OUT)
        # Save vectorizer via joblib as well for inspection
        joblib.dump(clf, MODEL_OUT)
        print('Saved TF-IDF pipeline model to', MODEL_OUT)

if __name__ == '__main__':
    main()
