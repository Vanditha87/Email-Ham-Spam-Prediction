import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as nb
st.image('Innomatics-Logo1.webp',width=200)
name = st.title('Email Spam and Ham Prediction')
model = pickle.load(open('model.pkl','rb'))
bow=pickle.load(open("bow.pkl",'rb'))
email = st.text_input('Enter the Email:')
if st.button("Submit"):
    data = bow.transform([email]).toarray() 
    spam_ham = model.predict(data)[0]
    spam_ham

    if spam_ham=='spam':
        st.image("download.png",width=200)
