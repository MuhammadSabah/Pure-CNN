"""
Activation functions for neural networks
All implemented from scratch using numpy
"""

import numpy as np


class Activation:
    """Base class for activation functions"""
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation function"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass: f(x) = max(0, x)"""
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        """Backward pass: derivative is 1 if x > 0, else 0"""
        dx = dout.copy()
        dx[self.cache <= 0] = 0
        return dx


class Sigmoid(Activation):
    """Sigmoid activation function"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass: f(x) = 1 / (1 + exp(-x))"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        output = 1 / (1 + np.exp(-x))
        self.cache = output
        return output
    
    def backward(self, dout):
        """Backward pass: f'(x) = f(x) * (1 - f(x))"""
        return dout * self.cache * (1 - self.cache)


class Tanh(Activation):
    """Hyperbolic tangent activation function"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass: f(x) = tanh(x)"""
        output = np.tanh(x)
        self.cache = output
        return output
    
    def backward(self, dout):
        """Backward pass: f'(x) = 1 - tanh²(x)"""
        return dout * (1 - self.cache ** 2)


class Softmax(Activation):
    """Softmax activation function (for multi-class classification)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass with numerical stability"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = output
        return output
    
    def backward(self, dout):
        """Backward pass for softmax"""
        return dout  # Simplified for cross-entropy loss 