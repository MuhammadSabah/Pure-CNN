"""
CNN Layer implementations from scratch using numpy
"""

import numpy as np
from .activations import ReLU, Sigmoid


class Layer:
    """Base class for all layers"""
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class ConvLayer(Layer):
    """Convolutional layer implementation"""
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0, input_channels=3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        
        # Initialize weights and biases
        self.weights = np.random.randn(num_filters, input_channels, filter_size, filter_size) * 0.1
        self.biases = np.zeros((num_filters, 1))
        
        # For backpropagation
        self.cache = None
        
    def forward(self, x):
        """
        Forward pass for convolution
        x shape: (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Add padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # Perform convolution
        for n in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        # Element-wise multiplication and sum
                        output[n, f, i, j] = np.sum(
                            x_padded[n, :, h_start:h_end, w_start:w_end] * self.weights[f, :, :, :]
                        ) + self.biases[f]
        
        # Cache for backpropagation
        self.cache = (x, x_padded)
        return output
    
    def backward(self, dout):
        """Backward pass for convolution"""
        x, x_padded = self.cache
        batch_size, channels, height, width = x.shape
        
        # Initialize gradients
        dx = np.zeros_like(x)
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)
        
        out_height, out_width = dout.shape[2], dout.shape[3]
        
        # Calculate gradients
        for n in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.filter_size
                        w_start = j * self.stride
                        w_end = w_start + self.filter_size
                        
                        # Gradient w.r.t weights
                        dw[f, :, :, :] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                        
                        # Gradient w.r.t input
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.weights[f, :, :, :] * dout[n, f, i, j]
        
        # Gradient w.r.t bias
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.num_filters, 1)
        
        # Remove padding from dx
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        # Update weights and biases (will be done by optimizer)
        self.dw = dw
        self.db = db
        
        return dx


class MaxPoolLayer(Layer):
    """Max pooling layer implementation"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass for max pooling
        x shape: (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform max pooling
        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        output[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])
        
        # Cache for backpropagation
        self.cache = x
        return output
    
    def backward(self, dout):
        """Backward pass for max pooling"""
        x = self.cache
        batch_size, channels, height, width = x.shape
        out_height, out_width = dout.shape[2], dout.shape[3]
        
        # Initialize gradient
        dx = np.zeros_like(x)
        
        # Distribute gradients to max elements
        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Find position of max element
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        
                        # Distribute gradient to max positions
                        dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
        
        return dx


class FullyConnectedLayer(Layer):
    """Fully connected (dense) layer implementation"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # For backpropagation
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass for fully connected layer
        x shape: (batch_size, input_size)
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # Linear transformation
        output = np.dot(x, self.weights) + self.biases
        
        # Cache for backpropagation
        self.cache = x
        return output
    
    def backward(self, dout):
        """Backward pass for fully connected layer"""
        x = self.cache
        
        # Gradients
        dx = np.dot(dout, self.weights.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        
        # Store gradients for optimizer
        self.dw = dw
        self.db = db
        
        return dx


class DropoutLayer(Layer):
    """Dropout layer for regularization"""
    
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """Forward pass for dropout"""
        if self.training and self.drop_rate > 0:
            self.mask = np.random.rand(*x.shape) > self.drop_rate
            return x * self.mask / (1 - self.drop_rate)
        else:
            return x
    
    def backward(self, dout):
        """Backward pass for dropout"""
        if self.training and self.mask is not None:
            return dout * self.mask / (1 - self.drop_rate)
        else:
            return dout
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training


class FlattenLayer(Layer):
    """Flatten layer to convert from conv to fully connected"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        """Flatten all dimensions except batch size"""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        """Reshape back to original dimensions"""
        return dout.reshape(self.input_shape) 