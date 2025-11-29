import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class SimpleDroneDetector(nn.Module):
    """Simple CNN model for drone detection (same as your training)"""
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ModelLoader:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['non_drone', 'drone']
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Initialize model
            self.model = SimpleDroneDetector(in_channels=3)
            
            # Try to load model from various possible locations
            model_paths = [
                'models/best_model.pth',
                'best_model.pth',
                '/app/models/best_model.pth',
                './best_model.pth'
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if Path(model_path).exists():
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    self.model.to(self.device)
                    self.model.eval()
                    model_loaded = True
                    logger.info(f"Model loaded successfully from {model_path}")
                    break
            
            if not model_loaded:
                logger.warning("No pre-trained model found. Using untrained model.")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, features, threshold=0.5):
        """Make prediction with adjustable threshold"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        try:
            # Convert to tensor
            if isinstance(features, np.ndarray):
                if features.ndim == 3:
                    features = np.expand_dims(features, 0)  # Add batch dimension
                
                features_tensor = torch.tensor(features, dtype=torch.float32)
            else:
                features_tensor = features
            
            # Move to device
            features_tensor = features_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert to Python types
                confidence = confidence.cpu().numpy()[0]
                predicted = predicted.cpu().numpy()[0]
                prob_np = probabilities.cpu().numpy()[0]

                is_drone = False
                
                # Use adjustable threshold
                is_drone = bool(prob_np[1] >= threshold)
                
                return {
                    'is_drone': is_drone,
                    'confidence': float(confidence),
                    'class_probabilities': {
                        'non_drone': float(prob_np[0]),
                        'drone': float(prob_np[1])
                    },
                    'predicted_class': self.class_names[predicted],
                    'raw_drone_probability': float(prob_np[1]),
                    'threshold_used': threshold
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, features_list):
        """Batch prediction for multiple feature sets"""
        results = []
        for features in features_list:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'status': 'error'})
        return results
    
def verify_model(self):
    """Verify the model is loaded and working"""
    if not self.model:
        return {"error": "Model not loaded"}
    
    # Test with random data
    test_input = torch.randn(1, 3, 64, 259)
    try:
        with torch.no_grad():
            output = self.model(test_input)
            return {
                "model_loaded": True,
                "test_output_shape": output.shape,
                "test_output_range": [float(output.min()), float(output.max())]
            }
    except Exception as e:
        return {"error": f"Model test failed: {str(e)}"}