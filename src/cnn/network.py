"""
Main CNN Network class that combines all layers
"""

import numpy as np
import pickle
from .layers import ConvLayer, MaxPoolLayer, FullyConnectedLayer, DropoutLayer, FlattenLayer
from .activations import ReLU, Sigmoid
from .optimizers import Adam


class CNN:
    """Convolutional Neural Network for binary classification"""
    
    def __init__(self, input_shape=(3, 64, 64), num_classes=1):
        """
        Initialize CNN architecture
        
        Args:
            input_shape: (channels, height, width) of input images
            num_classes: Number of output classes (1 for binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        self.training = True
        
        # Build the network architecture
        self._build_network()
    
    def _build_network(self):
        """Build the CNN architecture - BALANCED for accuracy and speed"""
        channels, height, width = self.input_shape
        
        # First convolutional block
        self.layers.append(ConvLayer(num_filters=32, filter_size=3, stride=1, padding=1, input_channels=channels))
        self.layers.append(ReLU())
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2))
        
        # Second convolutional block
        self.layers.append(ConvLayer(num_filters=64, filter_size=3, stride=1, padding=1, input_channels=32))
        self.layers.append(ReLU())
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2))
        
        # Flatten layer
        self.layers.append(FlattenLayer())
        
        # Calculate size after convolutions and pooling
        # Each pooling reduces size by factor of 2 (3 pooling layers)
        conv_output_size = 64 * (height // 4) * (width // 4)
        
        # Smaller fully connected layers for speed
        self.layers.append(FullyConnectedLayer(conv_output_size, 32))
        self.layers.append(ReLU())
        self.layers.append(DropoutLayer(drop_rate=0.3))
        self.layers.append(FullyConnectedLayer(32, self.num_classes))
        self.layers.append(Sigmoid())
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input batch of shape (batch_size, channels, height, width)
            
        Returns:
            Output predictions
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dout):
        """
        Backward pass through the network
        
        Args:
            dout: Gradient from loss function
        """
        gradient = dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def train_mode(self):
        """Set network to training mode"""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(True)
    
    def eval_mode(self):
        """Set network to evaluation mode"""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(False)
    
    def compute_loss(self, predictions, targets):
        """
        Compute binary cross-entropy loss
        
        Args:
            predictions: Model predictions of shape (batch_size, 1)
            targets: True labels of shape (batch_size, 1)
            
        Returns:
            loss: Scalar loss value
            dout: Gradient of loss w.r.t predictions
        """
        batch_size = predictions.shape[0]
        
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy loss
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
        # Gradient of loss w.r.t predictions
        dout = -(targets / predictions - (1 - targets) / (1 - predictions)) / batch_size
        
        return loss, dout
    
    def update_parameters(self, optimizer):
        """Update network parameters using optimizer"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                optimizer.update(layer)
    
    def predict(self, x):
        """
        Make predictions on input data
        
        Args:
            x: Input data
            
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities
        """
        self.eval_mode()
        probabilities = self.forward(x)
        predictions = (probabilities > 0.5).astype(int)
        self.train_mode()
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Save model parameters to file"""
        model_data = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'layers': []
        }
        
        for layer in self.layers:
            layer_data = {
                'type': type(layer).__name__
            }
            
            # Save layer parameters
            if hasattr(layer, 'weights'):
                layer_data['weights'] = layer.weights
            if hasattr(layer, 'biases'):
                layer_data['biases'] = layer.biases
            if hasattr(layer, 'num_filters'):
                layer_data['num_filters'] = layer.num_filters
                layer_data['filter_size'] = layer.filter_size
                layer_data['stride'] = layer.stride
                layer_data['padding'] = layer.padding
                layer_data['input_channels'] = layer.input_channels
            if hasattr(layer, 'pool_size'):
                layer_data['pool_size'] = layer.pool_size
                layer_data['stride'] = layer.stride
            if hasattr(layer, 'input_size'):
                layer_data['input_size'] = layer.input_size
                layer_data['output_size'] = layer.output_size
            if hasattr(layer, 'drop_rate'):
                layer_data['drop_rate'] = layer.drop_rate
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.input_shape = model_data['input_shape']
        self.num_classes = model_data['num_classes']
        
        # Rebuild network
        self._build_network()
        
        # Load parameters
        for i, layer_data in enumerate(model_data['layers']):
            layer = self.layers[i]
            
            if 'weights' in layer_data:
                layer.weights = layer_data['weights']
            if 'biases' in layer_data:
                layer.biases = layer_data['biases']
        
        print(f"Model loaded from {filepath}")
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        total_params = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total_params += layer.weights.size
            if hasattr(layer, 'biases'):
                total_params += layer.biases.size
        return total_params 