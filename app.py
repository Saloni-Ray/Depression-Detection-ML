import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('stopwords')
def transform_text(text):
    text = re.sub(r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]|\b\S+\.\S+\b", ' ', text)

    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    exclude_words = ["not", "won't", "shouldn't", "couldn't", "haven't", "can't", "aren't", "isn't", "don't", "doesn't",
                     "hasn't", "hadn't", "mightn't", "mustn't", "needn't", "shan't", "wasn't", "weren't", "wouldn't"]
    all_stopwords = stopwords.words('english')
    all_stopwords = [word for word in all_stopwords if word not in exclude_words]
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]

    return ' '.join(text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Depressive/Normal Text Classifier")

input_sms = st.text_area("Enter the text")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Depressive")
    else:
        st.header("Normal")
