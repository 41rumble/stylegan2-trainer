import React, { useState, useEffect } from 'react';
import { Container, Box, Typography, Paper, Button, TextField, Grid, LinearProgress, Alert, Switch, FormControlLabel } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { io } from 'socket.io-client';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const socket = io('http://localhost:54081');

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [trainingParams, setTrainingParams] = useState({
    learning_rate: 0.002,
    batch_size: 32,
    epochs: 100,
    image_size: 256,
    use_ada: true,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentMetrics, setCurrentMetrics] = useState({});
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');

  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to server');
    });

    socket.on('training_update', (metrics) => {
      setProgress(metrics.progress * 100);
      setCurrentMetrics(metrics);
    });

    socket.on('training_error', (data) => {
      setError(data.error);
      setIsTraining(false);
    });

    socket.on('training_complete', () => {
      setIsTraining(false);
      setProgress(100);
    });

    socket.on('new_samples', (data) => {
      setSamples((prevSamples) => [...prevSamples, data.path]);
    });

    return () => {
      socket.off('connect');
      socket.off('training_update');
      socket.off('training_error');
      socket.off('training_complete');
      socket.off('new_samples');
    };
  }, []);

  const handleFileChange = (event) => {
    setSelectedFiles(Array.from(event.target.files));
  };

  const handleParamChange = (param) => (event) => {
    setTrainingParams({
      ...trainingParams,
      [param]: Number(event.target.value),
    });
  };

  const handleUpload = async () => {
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('images[]', file);
    });

    try {
      setUploadStatus('uploading');
      const response = await fetch('http://localhost:54081/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setUploadStatus('success');
      console.log('Upload response:', data);
    } catch (error) {
      setUploadStatus('error');
      console.error('Upload error:', error);
    }
  };

  const startTraining = async () => {
    try {
      const response = await fetch('http://localhost:54081/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingParams),
      });
      const data = await response.json();
      setIsTraining(true);
      setError(null);
      setSamples([]);
      console.log('Training started:', data);
    } catch (error) {
      setError('Failed to start training');
      console.error('Training error:', error);
    }
  };

  const generateSamples = async () => {
    try {
      const response = await fetch('http://localhost:54081/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ num_samples: 4 }),
      });
      const data = await response.json();
      setSamples((prevSamples) => [...prevSamples, data.path]);
    } catch (error) {
      setError('Failed to generate samples');
      console.error('Generation error:', error);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            StyleGAN2 Training Interface
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Dataset Upload
                </Typography>
                <Button
                  variant="contained"
                  component="label"
                  sx={{ mb: 2 }}
                >
                  Select Images
                  <input
                    type="file"
                    hidden
                    multiple
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </Button>
                <Typography>
                  {selectedFiles.length} files selected
                </Typography>
                <Button
                  variant="contained"
                  onClick={handleUpload}
                  disabled={!selectedFiles.length || uploadStatus === 'uploading'}
                  sx={{ mt: 2 }}
                >
                  {uploadStatus === 'uploading' ? 'Uploading...' : 'Upload Dataset'}
                </Button>
                {uploadStatus === 'success' && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    Upload successful
                  </Alert>
                )}
              </Paper>

              <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Training Parameters
                </Typography>
                <TextField
                  fullWidth
                  label="Learning Rate"
                  type="number"
                  value={trainingParams.learning_rate}
                  onChange={handleParamChange('learning_rate')}
                  sx={{ mb: 2 }}
                  inputProps={{ step: 0.0001 }}
                />
                <TextField
                  fullWidth
                  label="Batch Size"
                  type="number"
                  value={trainingParams.batch_size}
                  onChange={handleParamChange('batch_size')}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Epochs"
                  type="number"
                  value={trainingParams.epochs}
                  onChange={handleParamChange('epochs')}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Image Size"
                  type="number"
                  value={trainingParams.image_size}
                  onChange={handleParamChange('image_size')}
                  sx={{ mb: 2 }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={trainingParams.use_ada}
                      onChange={(e) => setTrainingParams(prev => ({
                        ...prev,
                        use_ada: e.target.checked
                      }))}
                    />
                  }
                  label="Enable Adaptive Discriminator Augmentation (ADA)"
                  sx={{ mb: 2 }}
                />
                <Button
                  variant="contained"
                  onClick={startTraining}
                  disabled={isTraining || uploadStatus !== 'success'}
                  sx={{ mt: 2 }}
                >
                  {isTraining ? 'Training...' : 'Start Training'}
                </Button>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Training Progress
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={progress}
                  sx={{ mb: 2 }}
                />
                <Typography>
                  Progress: {progress.toFixed(1)}%
                </Typography>
                {currentMetrics.g_loss && (
                  <>
                    <Typography>
                      Generator Loss: {currentMetrics.g_loss.toFixed(4)}
                    </Typography>
                    <Typography>
                      Discriminator Loss: {currentMetrics.d_loss.toFixed(4)}
                    </Typography>
                    {currentMetrics.ada_p !== undefined && (
                      <>
                        <Typography>
                          ADA Probability: {(currentMetrics.ada_p * 100).toFixed(1)}%
                        </Typography>
                        <Typography>
                          Real-time Sign: {currentMetrics.ada_rt.toFixed(3)}
                        </Typography>
                      </>
                    )}
                  </>
                )}
              </Paper>

              <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Generated Samples
                </Typography>
                <Button
                  variant="contained"
                  onClick={generateSamples}
                  disabled={!currentMetrics.g_loss}
                  sx={{ mb: 2 }}
                >
                  Generate New Samples
                </Button>
                <Grid container spacing={2}>
                  {samples.map((sample, index) => (
                    <Grid item xs={6} key={index}>
                      <img
                        src={sample}
                        alt={`Generated sample ${index + 1}`}
                        style={{ width: '100%', height: 'auto' }}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
