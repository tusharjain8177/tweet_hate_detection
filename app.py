from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the tweet from the form
    tweet = request.form['tweet']
    
    # Extract features from the tweet
    features = vectorizer.transform([tweet])
    
    # Predict the label
    label = model.predict(features)[0]
    
    # Determine the class
    if label == 0:
        class_name = 'Not hate speech'
    else:
        class_name = 'Hate speech'
    
    return render_template('index.html', tweet=tweet, class_name=class_name)

if __name__ == '__main__':
    app.run()
