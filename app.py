from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load tokenizer and model
print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as file:
    tokenize = pickle.load(file)

print("Loading model...")
model = load_model('modelLSTMRNN.h5')
print("Model loaded successfully!")

def sentiment_analyze(sentence):
    """Analyze sentiment of a given sentence"""
    tokenized_sentence = tokenize.texts_to_sequences([sentence.lower()])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=300, padding='post')
    result = model.predict(padded_sentence, verbose=0)
    
    sentiment = "Positive" if result[0][0] > 0.5 else "Negative"
    confidence = float(result[0][0]) if result[0][0] > 0.5 else float(1 - result[0][0])
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "raw_score": float(result[0][0])
    }

@app.route('/')
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoints": {
            "/predict": "POST - Analyze sentiment of text",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Please provide 'text' in JSON body"
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Analyze sentiment
        result = sentiment_analyze(text)
        
        return jsonify({
            "success": True,
            "input_text": text,
            "result": result
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)