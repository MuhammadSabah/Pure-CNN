"""
Optimization algorithms for neural network training
"""

import numpy as np


class Optimizer:
    """Base class for optimizers"""
    
    def update(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, layer):
        """Update layer parameters using SGD with momentum"""
        layer_id = id(layer)
        
        # Initialize velocity if needed
        if layer_id not in self.velocity:
            self.velocity[layer_id] = {
                'weights': np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None,
                'biases': np.zeros_like(layer.biases) if hasattr(layer, 'biases') else None
            }
        
        # Update weights
        if hasattr(layer, 'weights') and hasattr(layer, 'dw'):
            if self.momentum > 0:
                self.velocity[layer_id]['weights'] = (
                    self.momentum * self.velocity[layer_id]['weights'] - 
                    self.learning_rate * layer.dw
                )
                layer.weights += self.velocity[layer_id]['weights']
            else:
                layer.weights -= self.learning_rate * layer.dw
        
        # Update biases
        if hasattr(layer, 'biases') and hasattr(layer, 'db'):
            if self.momentum > 0:
                self.velocity[layer_id]['biases'] = (
                    self.momentum * self.velocity[layer_id]['biases'] - 
                    self.learning_rate * layer.db
                )
                layer.biases += self.velocity[layer_id]['biases']
            else:
                layer.biases -= self.learning_rate * layer.db


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def update(self, layer):
        """Update layer parameters using Adam optimizer"""
        self.t += 1
        layer_id = id(layer)
        
        # Initialize moments if needed
        if layer_id not in self.m:
            self.m[layer_id] = {
                'weights': np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None,
                'biases': np.zeros_like(layer.biases) if hasattr(layer, 'biases') else None
            }
            self.v[layer_id] = {
                'weights': np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None,
                'biases': np.zeros_like(layer.biases) if hasattr(layer, 'biases') else None
            }
        
        # Update weights
        if hasattr(layer, 'weights') and hasattr(layer, 'dw'):
            # Update biased first moment estimate
            self.m[layer_id]['weights'] = (
                self.beta1 * self.m[layer_id]['weights'] + 
                (1 - self.beta1) * layer.dw
            )
            
            # Update biased second raw moment estimate
            self.v[layer_id]['weights'] = (
                self.beta2 * self.v[layer_id]['weights'] + 
                (1 - self.beta2) * (layer.dw ** 2)
            )
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[layer_id]['weights'] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[layer_id]['weights'] / (1 - self.beta2 ** self.t)
            
            # Update weights
            layer.weights -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        # Update biases
        if hasattr(layer, 'biases') and hasattr(layer, 'db'):
            # Update biased first moment estimate
            self.m[layer_id]['biases'] = (
                self.beta1 * self.m[layer_id]['biases'] + 
                (1 - self.beta1) * layer.db
            )
            
            # Update biased second raw moment estimate
            self.v[layer_id]['biases'] = (
                self.beta2 * self.v[layer_id]['biases'] + 
                (1 - self.beta2) * (layer.db ** 2)
            )
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[layer_id]['biases'] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[layer_id]['biases'] / (1 - self.beta2 ** self.t)
            
            # Update biases
            layer.biases -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, layer):
        """Update layer parameters using RMSprop"""
        layer_id = id(layer)
        
        # Initialize cache if needed
        if layer_id not in self.cache:
            self.cache[layer_id] = {
                'weights': np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None,
                'biases': np.zeros_like(layer.biases) if hasattr(layer, 'biases') else None
            }
        
        # Update weights
        if hasattr(layer, 'weights') and hasattr(layer, 'dw'):
            self.cache[layer_id]['weights'] = (
                self.decay_rate * self.cache[layer_id]['weights'] + 
                (1 - self.decay_rate) * (layer.dw ** 2)
            )
            layer.weights -= (
                self.learning_rate * layer.dw / 
                (np.sqrt(self.cache[layer_id]['weights']) + self.epsilon)
            )
        
        # Update biases
        if hasattr(layer, 'biases') and hasattr(layer, 'db'):
            self.cache[layer_id]['biases'] = (
                self.decay_rate * self.cache[layer_id]['biases'] + 
                (1 - self.decay_rate) * (layer.db ** 2)
            )
            layer.biases -= (
                self.learning_rate * layer.db / 
                (np.sqrt(self.cache[layer_id]['biases']) + self.epsilon)
            ) 