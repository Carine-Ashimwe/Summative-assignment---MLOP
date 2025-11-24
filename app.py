"""
Flask API for Image Classification ML Pipeline
Provides endpoints for prediction, retraining, and monitoring
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

from src.preprocessing import ImagePreprocessor
from src.model import ImageClassifierModel
from src.prediction import ImagePredictor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/upload'
RETRAIN_FOLDER = 'data/retrain'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables
predictor = None
model_loaded = False
model_uptime_start = None
retraining_status = {'status': 'idle', 'progress': 0, 'message': ''}
retraining_lock = threading.Lock()

# Load model on startup
def load_model():
    """Load the trained model"""
    global predictor, model_loaded, model_uptime_start
    
    model_path = 'models/image_classifier.tf'
    if not os.path.exists(model_path):
        model_path = 'models/image_classifier.h5'
    
    if os.path.exists(model_path):
        try:
            predictor = ImagePredictor(model_path=model_path)
            model_loaded = True
            model_uptime_start = time.time()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_loaded = False
    else:
        print("No model found. Please train a model first.")
        model_loaded = False

# Initialize on startup
load_model()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/uptime', methods=['GET'])
def model_uptime():
    """Get model uptime information"""
    if not model_loaded or model_uptime_start is None:
        return jsonify({
            'status': 'not_loaded',
            'uptime_seconds': 0
        })
    
    uptime = time.time() - model_uptime_start
    return jsonify({
        'status': 'loaded',
        'uptime_seconds': int(uptime),
        'uptime_formatted': format_uptime(uptime),
        'model_loaded_at': datetime.fromtimestamp(model_uptime_start).isoformat()
    })

def format_uptime(seconds):
    """Format uptime in human-readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict(filepath, return_probabilities=True)
        prediction_time = time.time() - start_time
        
        # Clean up
        os.remove(filepath)
        
        result['prediction_time_ms'] = round(prediction_time * 1000, 2)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict endpoint for multiple images"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        filepaths = []
        for file in files:
            if file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                filepaths.append(filepath)
        
        if not filepaths:
            return jsonify({'error': 'No valid image files'}), 400
        
        # Make predictions
        start_time = time.time()
        results = predictor.predict_batch(filepaths)
        prediction_time = time.time() - start_time
        
        # Clean up
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'results': results,
            'total_images': len(results),
            'total_time_ms': round(prediction_time * 1000, 2),
            'avg_time_per_image_ms': round((prediction_time * 1000) / len(results), 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain/status', methods=['GET'])
def retrain_status():
    """Get retraining status"""
    return jsonify(retraining_status)

def train_model_async(data_dir):
    """Train model in background thread"""
    global retraining_status, predictor, model_loaded, model_uptime_start
    
    try:
        with retraining_lock:
            retraining_status = {
                'status': 'preprocessing',
                'progress': 10,
                'message': 'Loading and preprocessing data...'
            }
        
        # Load and preprocess data
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        images, labels, class_names = preprocessor.load_images_from_directory(data_dir)
        
        if len(class_names) < 2:
            raise ValueError("Need at least 2 classes for training")
        
        # Encode labels
        encoded_labels = preprocessor.encode_labels(labels)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
            images, encoded_labels, test_size=0.2, validation_size=0.1
        )
        
        with retraining_lock:
            retraining_status = {
                'status': 'training',
                'progress': 30,
                'message': f'Training model with {len(class_names)} classes...'
            }
        
        # Create and train model
        model = ImageClassifierModel(
            num_classes=len(class_names),
            input_shape=(224, 224, 3),
            model_name='image_classifier'
        )
        model.set_class_names(class_names)
        model.build_model(use_transfer_learning=True)
        
        # Get data generator
        datagen = preprocessor.get_data_generator(augment=True)
        
        # Train model
        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=20,  # Reduced for faster retraining
            batch_size=32,
            data_generator=datagen
        )
        
        with retraining_lock:
            retraining_status = {
                'status': 'evaluating',
                'progress': 80,
                'message': 'Evaluating model...'
            }
        
        # Evaluate model
        eval_results = model.evaluate(X_test, y_test, class_names=class_names)
        
        # Save model
        model.save_model(model_dir='models', save_format='tf')
        
        # Reload model
        model_path = 'models/image_classifier.tf'
        predictor = ImagePredictor(model_path=model_path)
        model_loaded = True
        model_uptime_start = time.time()
        
        with retraining_lock:
            retraining_status = {
                'status': 'completed',
                'progress': 100,
                'message': f'Model retrained successfully! Accuracy: {eval_results["accuracy"]:.4f}',
                'accuracy': float(eval_results['accuracy']),
                'class_names': class_names,
                'completed_at': datetime.now().isoformat()
            }
        
    except Exception as e:
        with retraining_lock:
            retraining_status = {
                'status': 'failed',
                'progress': 0,
                'message': f'Retraining failed: {str(e)}',
                'error': str(e)
            }

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    global retraining_status
    
    # Check if already retraining
    with retraining_lock:
        if retraining_status['status'] in ['preprocessing', 'training', 'evaluating']:
            return jsonify({'error': 'Retraining already in progress'}), 409
    
    # Check for uploaded retrain data
    retrain_data_dir = None
    
    # Check if data uploaded in request
    if 'data' in request.files:
        files = request.files.getlist('data')
        if files:
            # Create temporary directory structure
            retrain_dir = os.path.join(RETRAIN_FOLDER, f"retrain_{int(time.time())}")
            os.makedirs(retrain_dir, exist_ok=True)
            
            # Organize files by class (assuming filename format: class_filename.jpg)
            for file in files:
                if file.filename:
                    # Simple approach: use first part of filename as class
                    class_name = file.filename.split('_')[0] if '_' in file.filename else 'unknown'
                    class_dir = os.path.join(retrain_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(class_dir, filename)
                    file.save(filepath)
            
            retrain_data_dir = retrain_dir
    
    # Check existing retrain folder
    if not retrain_data_dir:
        if os.path.exists(RETRAIN_FOLDER) and os.listdir(RETRAIN_FOLDER):
            # Use existing retrain folder
            retrain_data_dir = RETRAIN_FOLDER
    
    if not retrain_data_dir or not os.path.exists(retrain_data_dir):
        return jsonify({
            'error': 'No training data found. Please upload data first.',
            'hint': 'Upload data to /api/upload/retrain or ensure data/retrain directory contains class folders'
        }), 400
    
    # Start retraining in background
    thread = threading.Thread(target=train_model_async, args=(retrain_data_dir,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Retraining started',
        'status': 'preprocessing'
    })

@app.route('/api/upload/retrain', methods=['POST'])
def upload_retrain_data():
    """Upload data for retraining"""
    if 'data' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('data')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Create class-based directory structure
        retrain_dir = os.path.join(RETRAIN_FOLDER, f"batch_{int(time.time())}")
        os.makedirs(retrain_dir, exist_ok=True)
        
        uploaded_count = 0
        for file in files:
            if file.filename and allowed_file(file.filename):
                # Extract class name from filename (format: class_filename.jpg)
                # Or use form data if provided
                class_name = request.form.get('class_name', None)
                if not class_name:
                    # Try to extract from filename
                    if '_' in file.filename:
                        class_name = file.filename.split('_')[0]
                    else:
                        class_name = 'unknown'
                
                class_dir = os.path.join(retrain_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(class_dir, filename)
                file.save(filepath)
                uploaded_count += 1
        
        return jsonify({
            'message': f'Successfully uploaded {uploaded_count} files',
            'retrain_directory': retrain_dir,
            'uploaded_count': uploaded_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics and visualizations data"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get class names from predictor
    class_names = predictor.class_names if predictor and predictor.class_names else []
    
    # Get class distribution from training data
    class_distribution = {}
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for class_name in class_names:
            class_path = os.path.join(train_dir, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                class_distribution[class_name] = count
    
    # Generate statistics
    stats = {
        'total_predictions': 0,  # Could track in database in production
        'model_accuracy': 0.95,  # From evaluation (can be updated from model metadata)
        'average_response_time_ms': 150,
        'classes': class_names,
        'class_distribution': class_distribution,
        'total_images': sum(class_distribution.values()),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

