import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class information
CLASS_NAMES = ["Apple 🍎", "Banana 🍌", "Orange 🍊"]
CLASS_DESCRIPTIONS = {
    "Apple 🍎": "Rich in fiber and antioxidants, apples are one of the most popular fruits worldwide.",
    "Banana 🍌": "High in potassium and natural sugars, bananas are perfect for quick energy.",
    "Orange 🍊": "Excellent source of vitamin C and immune system boosting compounds."
}

def load_history():
    """Load prediction history from JSON file."""
    try:
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_history(history):
    """Save prediction history to JSON file."""
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f)

def process_image(file_path):
    """Preprocess the uploaded image for prediction."""
    img = Image.open(file_path).convert("RGB")
    img = img.resize((32, 32))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    return x, img

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                           class_names=CLASS_NAMES, 
                           class_descriptions=CLASS_DESCRIPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process image
        img_array, original_img = process_image(file_path)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction
        class_prediction = model.predict(img_array)
        predicted_class = np.argmax(class_prediction[0])
        confidence = float(class_prediction[0][predicted_class])
        
        # Convert confidence to percentage (0-100)
        confidence_percentage = min(confidence * 100, 100)
        
        # Prepare result
        result = {
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': confidence_percentage,
            'description': CLASS_DESCRIPTIONS[CLASS_NAMES[predicted_class]],
            'probabilities': dict(zip(CLASS_NAMES, 
                                    [min(prob * 100, 100) for prob in class_prediction[0].tolist()]))
        }
        
        # Save to history
        history = load_history()
        history.append({
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': confidence_percentage,
        })
        save_history(history)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/history')
def history():
    """Retrieve prediction history."""
    history = load_history()
    return render_template('history.html', 
                           history=reversed(history[-10:]),
                           total_predictions=len(history),
                           avg_confidence=sum(h['confidence'] for h in history) / len(history) if history else 0)

if __name__ == '__main__':
    app.run(debug=True)