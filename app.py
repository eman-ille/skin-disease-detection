from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
from flask_cors import CORS

app = Flask(__name__, template_folder='.')
CORS(app) # Taake frontend aur backend bina kisi error ke baat kar sakein

import tf_keras as keras
# 1. Model Load Karna (Viva Point: H5 file binary format mein model save karti hai)
MODEL_PATH = 'skin_model_best.h5'
if os.path.exists(MODEL_PATH):
    # Standalone Keras 3 use kar rahe hain taake naye model formats support hon
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model {MODEL_PATH} loaded successfully using Keras 3!")
else:
    print(f"WARNING: Model file '{MODEL_PATH}' NOT FOUND. Backend will not be able to predict.")
    model = None

# 2. Class Names (Notebook ke mutabiq)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

@app.route('/')
def index():
    return render_template('code.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    if model is None:
        return jsonify({'error': 'AI Model not found on server'}), 500

    # 3. Image Preprocessing (Viva Point: Model 128x128 images par train hua tha)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0 # Normalization (0-1 range)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Prediction
    predictions = model.predict(img_array)[0]
    
    # Results ko format karna
    results = []
    for i in range(len(CLASS_NAMES)):
        results.append({
            'code': CLASS_NAMES[i],
            'name': FULL_NAMES[CLASS_NAMES[i]],
            'probability': float(predictions[i])
        })
    
    # Highest probability ke hisab se sort karna
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    return jsonify(results)

# --- TEXT-BASED DATABASE LOGIC (JSON) ---
import json
from datetime import datetime

DATA_FILE = 'patient_records.json'

def load_records():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_records(records):
    with open(DATA_FILE, 'w') as f:
        json.dump(records, f, indent=4)

@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.json
    if not data or 'patientName' not in data or 'diagnosis' not in data:
        return jsonify({'error': 'Invalid data'}), 400
    
    # Naya record tayyar karna
    new_record = {
        'id': 'P-' + str(int(datetime.now().timestamp()) % 10000),
        'patientName': data['patientName'],
        'diagnosis': data['diagnosis'],
        'confidence': data['confidence'],
        'date': datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
    
    # Purane records load karo, naya daalo, aur dobara save karo
    records = load_records()
    records.append(new_record)
    save_records(records)
    
    return jsonify({'success': True, 'record': new_record})

@app.route('/get_records', methods=['GET'])
def get_records():
    records = load_records()
    # Latest records pehle show honge
    return jsonify(list(reversed(records)))

@app.route('/delete_record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    records = load_records()
    # Filter out the record to delete
    records = [r for r in records if r.get('id') != record_id]
    save_records(records)
    return jsonify({'success': True})

if __name__ == '__main__':
    # Flask app run karna
    print("DermAI Backend starting on http://localhost:5000")
    app.run(debug=True, port=5000)
