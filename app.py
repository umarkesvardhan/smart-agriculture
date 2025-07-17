from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import cv2

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
try:
    with open('models/soil_model.pkl', 'rb') as f:
        soil_model = pickle.load(f)

    with open('models/soil_scaler.pkl', 'rb') as f:
        soil_scaler = pickle.load(f)

    with open('models/soil_types.pkl', 'rb') as f:
        soil_types = pickle.load(f)

    disease_model = load_model('models/disease_model_best.h5')

    with open('models/disease_classes.pkl', 'rb') as f:
        disease_classes = pickle.load(f)

    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise SystemExit("Could not start application due to model loading failure")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/soil-analysis', methods=['GET', 'POST'])
def soil_analysis():
    if request.method == 'POST':
        try:
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            ph = float(request.form['ph'])

            input_data = np.array([[n, p, k, ph]])
            scaled_data = soil_scaler.transform(input_data)
            prediction = soil_model.predict(scaled_data)
            soil_code = prediction[0]
            soil_type = soil_types.get(soil_code, 'Unknown')

            return render_template('soil_result.html', result={
                'soil_type': soil_type,
                'input_values': {'N': n, 'P': p, 'K': k, 'pH': ph}
            })

        except Exception as e:
            logger.error(f"Soil analysis failed: {e}")
            return render_template('error.html', message="Error during soil analysis.")

    return render_template('soil_form.html')

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                img = preprocess_image(filepath)
                predictions = disease_model.predict(img)
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])
                class_name = disease_classes[class_idx]

                return render_template('disease_result.html', result={
                    'image_path': filepath,
                    'disease': class_name.replace('_', ' ').title(),
                    'confidence': f"{confidence * 100:.2f}%",
                    'is_healthy': 'healthy' in class_name.lower()
                })
            except Exception as e:
                logger.error(f"Disease detection failed: {e}")
                return render_template('error.html', message="Error during disease detection.")

    return render_template('disease_form.html')

# API Routes
@app.route('/api/soil', methods=['POST'])
def api_soil():
    try:
        data = request.get_json()
        values = [data[key] for key in ['nitrogen', 'phosphorus', 'potassium', 'ph']]
        input_data = np.array([values])
        scaled_data = soil_scaler.transform(input_data)
        prediction = soil_model.predict(scaled_data)
        soil_code = prediction[0]
        soil_type = soil_types.get(soil_code, 'Unknown')
        return jsonify({'soil_type': soil_type, 'code': int(soil_code)})
    except Exception as e:
        logger.error(f"API soil analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/disease', methods=['POST'])
def api_disease():
    try:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = preprocess_image(filepath)
            predictions = disease_model.predict(img)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            class_name = disease_classes[class_idx]
            os.remove(filepath)
            return jsonify({
                'disease': class_name,
                'confidence': confidence,
                'is_healthy': 'healthy' in class_name.lower()
            })
        return jsonify({'error': 'Invalid or no file uploaded'}), 400
    except Exception as e:
        logger.error(f"API disease detection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

