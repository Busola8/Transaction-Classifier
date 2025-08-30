import streamlit as st, os, requests
st.title('Transaction Classifier Demo')

API = st.text_input('API URL', value='http://localhost:8000')
ts = st.text_input('Timestamp (ISO)', value='2025-08-01T10:30:00')
amount = st.number_input('Amount', value=1500.0)
merchant = st.text_input('Merchant', value='KFC')
description = st.text_input('Description', value='KFC Lagos POS 1234')
channel = st.selectbox('Channel', ['POS','USSD','MOBILE','ATM','WEB','POS_ONLINE'])

if st.button('Predict'):
    payload = {'timestamp': ts, 'amount': float(amount), 'merchant': merchant, 'description': description, 'channel': channel}
    resp = requests.post(API + '/predict', json=payload)
    if resp.ok:
        data = resp.json()
        st.write('Predicted label:', data.get('label'))
        st.write('Probabilities:', data.get('probs'))
    else:
        st.error(resp.text)
