from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
import re
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Load the tokenizer from file
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Access the stop words using scikit-learn
stop = ENGLISH_STOP_WORDS

# Define the preprocessing function
def preprocess_text(text):
    clean_text = text.lower()
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    clean_text = " ".join([word for word in clean_text.split() if word not in stop])
    return clean_text

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['message']

        # Preprocess the input text
        clean_text = preprocess_text(text)

        # Fit and transform the sequence
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

        # Make a prediction
        prediction = model.predict(np.array(padded_sequence))
        result = "FAKE" if prediction > 0.5 else "REAL"

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
