from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model and vectorizer at startup
model_path = 'fake_news_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
else:
    print("Error: Model or vectorizer not found. Please run train_model.py first.")
    clf = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not clf or not vectorizer:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400

    text = data['text']
    
    # Vectorize the input text
    try:
        text_vectorized = vectorizer.transform([text])
        prediction = clf.predict(text_vectorized)[0]
        
        # LinearSVC output: 0 -> REAL, 1 -> FAKE
        result_label = "REAL" if prediction == 0 else "FAKE"
        
        # Calculate a pseudo-confidence score using decision_function
        dist = abs(clf.decision_function(text_vectorized)[0])
        # Simple sigmoid-like normalization for confidence visualization
        import math
        confidence = 1 / (1 + math.exp(-dist))
        
        return jsonify({
            'prediction': result_label,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
