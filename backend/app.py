from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import json
import torch
import torchvision.utils as vutils
from models.stylegan2 import StyleGAN2Trainer
import threading
import time

app = Flask(__name__, static_folder='../static')
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = os.path.join(app.root_path, '..', 'static', 'uploads')
MODELS_FOLDER = os.path.join(app.root_path, '..', 'static', 'models')
SAMPLES_FOLDER = os.path.join(app.root_path, '..', 'static', 'samples')

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, SAMPLES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for training state
current_trainer = None
training_thread = None
is_training = False
training_progress = 0
current_metrics = {}

def training_callback(metrics):
    global training_progress, current_metrics
    training_progress = metrics['progress']
    current_metrics = metrics
    socketio.emit('training_update', metrics)
    
    # Generate and save sample images every 10% progress
    if int(training_progress * 10) > int((training_progress - 0.1) * 10):
        samples = current_trainer.generate_samples(4)
        sample_grid = vutils.make_grid(samples, nrow=2, normalize=True)
        sample_path = os.path.join(SAMPLES_FOLDER, f'samples_{int(training_progress * 100)}.png')
        vutils.save_image(sample_grid, sample_path)
        socketio.emit('new_samples', {'path': f'/static/samples/samples_{int(training_progress * 100)}.png'})

def train_model(params):
    global is_training, current_trainer, training_progress
    
    try:
        current_trainer = StyleGAN2Trainer(
            UPLOAD_FOLDER,
            image_size=params.get('image_size', 256),
            batch_size=params.get('batch_size', 32),
            lr=params.get('learning_rate', 0.002)
        )
        
        current_trainer.train(
            num_epochs=params.get('epochs', 100),
            callback=training_callback
        )
        
        # Save the trained model
        model_path = os.path.join(MODELS_FOLDER, 'latest_model.pth')
        current_trainer.save_model(model_path)
        
    except Exception as e:
        socketio.emit('training_error', {'error': str(e)})
    finally:
        is_training = False
        training_progress = 1.0
        socketio.emit('training_complete')

@app.route('/api/upload', methods=['POST'])
def upload_images():
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images[]')
    saved_files = []
    
    for file in files:
        if file.filename:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            saved_files.append(file.filename)
    
    return jsonify({'message': 'Files uploaded successfully', 'files': saved_files})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global is_training, training_thread
    
    if is_training:
        return jsonify({'error': 'Training already in progress'}), 400
    
    params = request.json
    is_training = True
    training_progress = 0
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model, args=(params,))
    training_thread.start()
    
    return jsonify({'message': 'Training started', 'params': params})

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    global is_training, training_progress, current_metrics
    
    return jsonify({
        'is_training': is_training,
        'progress': training_progress,
        'metrics': current_metrics
    })

@app.route('/api/generate', methods=['POST'])
def generate_samples():
    if not current_trainer:
        return jsonify({'error': 'No trained model available'}), 400
    
    num_samples = request.json.get('num_samples', 4)
    samples = current_trainer.generate_samples(num_samples)
    
    # Save samples
    sample_grid = vutils.make_grid(samples, nrow=2, normalize=True)
    sample_path = os.path.join(SAMPLES_FOLDER, f'generated_{int(time.time())}.png')
    vutils.save_image(sample_grid, sample_path)
    
    return jsonify({
        'message': 'Samples generated successfully',
        'path': f'/static/samples/generated_{int(time.time())}.png'
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    if is_training:
        socketio.emit('training_update', current_metrics)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=54081, debug=True, allow_unsafe_werkzeug=True)