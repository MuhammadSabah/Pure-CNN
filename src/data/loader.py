"""
Data loading utilities for cat vs dog classification
"""

import numpy as np
import os
from .preprocessing import ImagePreprocessor


class DataLoader:
    """Data loader for cat vs dog classification dataset"""
    
    def __init__(self, cats_dir, dogs_dir, batch_size=32, target_size=(64, 64), 
                 validation_split=0.2, max_images_per_class=None):
        """
        Initialize data loader
        
        Args:
            cats_dir: Directory containing cat images
            dogs_dir: Directory containing dog images
            batch_size: Batch size for training
            target_size: Target image size (width, height)
            validation_split: Fraction of data to use for validation
            max_images_per_class: Maximum images per class (None for all)
        """
        self.cats_dir = cats_dir
        self.dogs_dir = dogs_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        
        # Load and prepare data
        self.train_images, self.train_labels = None, None
        self.val_images, self.val_labels = None, None
        self.load_data(max_images_per_class)
    
    def load_data(self, max_images_per_class=None):
        """Load and split data into training and validation sets"""
        print("Loading cat images...")
        cat_images, cat_paths = self.preprocessor.load_images_from_directory(
            self.cats_dir, max_images_per_class
        )
        
        print("Loading dog images...")
        dog_images, dog_paths = self.preprocessor.load_images_from_directory(
            self.dogs_dir, max_images_per_class
        )
        
        if len(cat_images) == 0 or len(dog_images) == 0:
            raise ValueError("Could not load images. Check directory paths.")
        
        # Create labels (0 for cats, 1 for dogs)
        cat_labels = np.zeros((len(cat_images), 1))
        dog_labels = np.ones((len(dog_images), 1))
        
        # Combine data
        all_images = np.concatenate([cat_images, dog_images], axis=0)
        all_labels = np.concatenate([cat_labels, dog_labels], axis=0)
        
        # Shuffle data
        indices = np.random.permutation(len(all_images))
        all_images = all_images[indices]
        all_labels = all_labels[indices]
        
        # Split into train and validation
        split_idx = int(len(all_images) * (1 - self.validation_split))
        
        self.train_images = all_images[:split_idx]
        self.train_labels = all_labels[:split_idx]
        self.val_images = all_images[split_idx:]
        self.val_labels = all_labels[split_idx:]
        
        print(f"Loaded {len(self.train_images)} training images and {len(self.val_images)} validation images")
        print(f"Training: {np.sum(self.train_labels == 0)} cats, {np.sum(self.train_labels == 1)} dogs")
        print(f"Validation: {np.sum(self.val_labels == 0)} cats, {np.sum(self.val_labels == 1)} dogs")
    
    def get_batch(self, batch_type='train'):
        """
        Generator for batches of data
        
        Args:
            batch_type: 'train' or 'val'
            
        Yields:
            batch_images, batch_labels: batches of images and labels
        """
        if batch_type == 'train':
            images, labels = self.train_images, self.train_labels
        else:
            images, labels = self.val_images, self.val_labels
        
        num_samples = len(images)
        indices = np.arange(num_samples)
        
        # Shuffle for training
        if batch_type == 'train':
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            
            yield batch_images, batch_labels
    
    def get_random_batch(self, batch_type='train'):
        """Get a random batch of data"""
        if batch_type == 'train':
            images, labels = self.train_images, self.train_labels
        else:
            images, labels = self.val_images, self.val_labels
        
        num_samples = len(images)
        batch_indices = np.random.choice(num_samples, size=min(self.batch_size, num_samples), replace=False)
        
        return images[batch_indices], labels[batch_indices]
    
    def get_full_dataset(self, batch_type='train'):
        """Get the full dataset"""
        if batch_type == 'train':
            return self.train_images, self.train_labels
        else:
            return self.val_images, self.val_labels
    
    def get_num_batches(self, batch_type='train'):
        """Get number of batches in dataset"""
        if batch_type == 'train':
            return len(self.train_images) // self.batch_size + (1 if len(self.train_images) % self.batch_size != 0 else 0)
        else:
            return len(self.val_images) // self.batch_size + (1 if len(self.val_images) % self.batch_size != 0 else 0)
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'train_size': len(self.train_images),
            'val_size': len(self.val_images),
            'input_shape': self.train_images.shape[1:],
            'batch_size': self.batch_size,
            'train_batches': self.get_num_batches('train'),
            'val_batches': self.get_num_batches('val')
        } 