import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

# Load the pre-trained model and vectorizer
tuned_and_lemma_svm_model = joblib.load('tuned_and_lemma_svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create Stopwords list
stopwords_list = stopwords.words('english') + list(string.punctuation) + list(map(str, range(10)))
stopwords_list += ['sxsw', 'mention', 'sxsw', 'link', 'link', 'rt', '#sxsw']

def preprocess_tweet(tweet, stopwords_list):
    tweet = tweet.lower()
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)
    tokens_without_stopwords = [str(i) for i in tokens if str(i) not in stopwords_list]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_without_stopwords]
    lemmatized_tweet = ' '.join(lemmatized_tokens)
    return lemmatized_tweet

def predict_sentiment(tweet):
    # Preprocess the input tweet
    preprocessed_tweet = preprocess_tweet(tweet, stopwords_list)

    # Vectorize the preprocessed tweet
    tweet_vectorized = tfidf_vectorizer.transform([preprocessed_tweet])

    # Convert the vectorized tweet to a dense array and then to a string
    tweet_vectorized_dense = np.array(tweet_vectorized.todense())
    tweet_vectorized_str = ' '.join(map(str, tweet_vectorized_dense))

    # Make a prediction
    prediction_encoded = tuned_and_lemma_svm_model.predict([tweet_vectorized_str])

    # Take the first element of the prediction array
    prediction = prediction_encoded[0]

    return prediction

# Streamlit UI
st.title('Sentiment Analysis')

tweet = st.text_input('Enter a tweet:')

if st.button('Predict'):
    if tweet:
        st.write(tweet)
        prediction = predict_sentiment(tweet)
        st.write(f'Prediction: {prediction}')
    else:
        st.warning('Please enter a tweet for prediction.')
