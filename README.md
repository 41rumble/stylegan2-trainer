# StyleGAN2 Training Interface

A web-based application for training StyleGAN2 models with real-time monitoring and visualization.

## Features

- User-friendly interface for uploading training images
- Configurable training parameters
- Real-time training progress monitoring
- Live display of generated samples
- Dark mode UI using Material-UI

## Architecture

### Backend (Flask + StyleGAN2)
- RESTful API endpoints for uploading images, starting training, and generating samples
- Real-time training progress updates using WebSocket
- Basic StyleGAN2 implementation with customizable parameters
- Support for saving and loading models
- Sample generation during and after training

### Frontend (React)
- Modern React application with Material-UI
- Real-time updates using Socket.IO
- Progress tracking and visualization
- Image upload and management
- Training parameter configuration

## Setup

### Backend

#### CPU Installation (Default)
1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

#### GPU Installation (Recommended for Training)
1. First, ensure you have CUDA installed on your system. You can check your CUDA version with:
```bash
nvidia-smi
```

2. Install PyTorch with CUDA support:
```bash
cd backend
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

Note: If you encounter CUDA-related errors, try the following:
1. Make sure your NVIDIA drivers are up to date
2. Check that your CUDA version matches the PyTorch CUDA version
3. If using Anaconda, create a new environment to avoid conflicts:
```bash
conda create -n stylegan2 python=3.11
conda activate stylegan2
# Then proceed with the installation steps above
```

#### Starting the Server
```bash
python app.py
```

The backend server will run on http://localhost:54081.

Note: Training on CPU will be significantly slower than on GPU. For best results, use a CUDA-capable GPU.

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend application will run on http://localhost:59843.

## Usage

1. Upload Dataset:
   - Click "Select Images" to choose your training images
   - Click "Upload Dataset" to upload the selected images

2. Configure Training:
   - Set the learning rate (default: 0.002)
   - Set the batch size (default: 32)
   - Set the number of epochs (default: 100)
   - Set the image size (default: 256)

3. Start Training:
   - Click "Start Training" to begin the training process
   - Monitor progress and losses in real-time
   - View generated samples as they are created

4. Generate Samples:
   - Click "Generate New Samples" to create new images from the trained model

## Project Structure

```
stylegan2_trainer/
├── backend/
│   ├── models/
│   │   └── stylegan2.py
│   ├── templates/
│   ├── utils/
│   ├── app.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.js
│   │   └── ...
│   ├── package.json
│   └── ...
└── static/
    ├── uploads/
    ├── models/
    └── samples/
```

## Future Enhancements

1. User authentication and project management
2. More advanced StyleGAN2 features and optimizations
3. Model comparison and versioning
4. Cloud GPU integration
5. Advanced dataset preprocessing options
6. Training checkpoints and resumption
7. More detailed training metrics and visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.