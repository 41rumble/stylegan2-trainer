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

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the server:
```bash
python app.py
```

The backend server will run on http://localhost:54081.

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