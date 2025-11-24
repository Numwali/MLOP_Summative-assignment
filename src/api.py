"""
Flask REST API for CIFAR-10 model
Provides endpoints for prediction, retraining, stats, and health monitoring.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from datetime import datetime
from werkzeug.utils import secure_filename
from src.prediction import CIFAR10Predictor
from src.train import retrain_model
import threading

app = Flask(**name**)
CORS(app)

# Configuration

UPLOAD_FOLDER = 'data/retrain'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize predictor

predictor = CIFAR10Predictor()

# Track API stats

api_stats = {
'start_time': datetime.now().isoformat(),
'total_predictions': 0,
'total_retrains': 0,
'last_prediction_time': None,
'average_response_time': 0
}
response_times = []

def allowed_file(filename):
"""Check if file extension is allowed"""
return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
"""Health check endpoint"""
uptime = (datetime.now() - datetime.fromisoformat(api_stats['start_time'])).total_seconds()
return jsonify({
'status': 'healthy',
'model_loaded': True,
'uptime_seconds': uptime,
'timestamp': datetime.now().isoformat()
})

@app.route('/predict', methods=['POST'])
def predict():
"""Predict image class"""
start_time = time.time()

```
if 'image' not in request.files:
    return jsonify({'error': 'No image provided'}), 400

file = request.files['image']
if file.filename == '':
    return jsonify({'error': 'No selected file'}), 400

if file and allowed_file(file.filename):
    try:
        image_bytes = file.read()
        result = predictor.predict_single(image_bytes)
        
        # Update API stats
        response_time = time.time() - start_time
        response_times.append(response_time)
        api_stats['total_predictions'] += 1
        api_stats['last_prediction_time'] = datetime.now().isoformat()
        api_stats['average_response_time'] = sum(response_times) / len(response_times)
        
        result['response_time'] = response_time
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
return jsonify({'error': 'Invalid file type'}), 400
```

@app.route('/upload-data', methods=['POST'])
def upload_training_data():
"""Upload images for retraining"""
if 'files' not in request.files:
return jsonify({'error': 'No files provided'}), 400

```
files = request.files.getlist('files')
class_name = request.form.get('class_name', 'unknown')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
if class_name not in class_names:
    return jsonify({'error': 'Invalid class name'}), 400

# Create directory for class
class_dir = os.path.join(app.config['UPLOAD_FOLDER'], class_name)
os.makedirs(class_dir, exist_ok=True)

uploaded_files = []
for file in files:
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(class_dir, filename)
        file.save(filepath)
        uploaded_files.append(filename)

return jsonify({
    'message': f'Uploaded {len(uploaded_files)} files',
    'files': uploaded_files,
    'class': class_name
}), 200
```

@app.route('/retrain', methods=['POST'])
def trigger_retrain():
"""Trigger model retraining with uploaded data"""
try:
def retrain_async():
retrain_model(app.config['UPLOAD_FOLDER'], epochs=20)
predictor.reload_model()
api_stats['total_retrains'] += 1

```
    thread = threading.Thread(target=retrain_async)
    thread.start()
    
    return jsonify({
        'message': 'Retraining started in background',
        'timestamp': datetime.now().isoformat()
    }), 202
except Exception as e:
    return jsonify({'error': str(e)}), 500
```

@app.route('/stats', methods=['GET'])
def get_stats():
"""Get API statistics and training metrics"""
metrics = {}
if os.path.exists('logs/training_logs.json'):
with open('logs/training_logs.json', 'r') as f:
metrics = json.load(f)

```
return jsonify({
    'api_stats': api_stats,
    'model_metrics': metrics,
    'uptime_seconds': (datetime.now() - datetime.fromisoformat(api_stats['start_time'])).total_seconds()
}), 200
```

@app.route('/model-info', methods=['GET'])
def model_info():
"""Get model information"""
return jsonify({
'model_type': 'CNN',
'framework': 'TensorFlow/Keras',
'input_shape': [32, 32, 3],
'num_classes': 10,
'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
}), 200

if **name** == '**main**':
os.makedirs('data/retrain', exist_ok=True)
os.makedirs('logs', exist_ok=True)
app.run(host='0.0.0.0', port=5000, debug=False)

