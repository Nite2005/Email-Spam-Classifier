import streamlit as st
import pickle
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower() #convert to lower
    text = nltk.word_tokenize(text)#split into words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)


    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y) 




cv = pickle.load(open('cv.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



st.title('Email Spam classifier')
input_sms = st.text_area("Enter the message")


    

if st.button('Predict'):
  #1. preprocess
  transform_sms = transform_text(input_sms)
  #2. vectorize


  vector_input  = cv.transform([transform_sms]).toarray()
  #3. predict
  result  = model.predict(vector_input)[0]
  st.header(result)
  #4.Display
  if result == 1:
      st.header("spam")
  else:
      st.header("Not Spam")

