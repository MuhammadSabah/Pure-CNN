"""
Image preprocessing utilities
"""

import numpy as np
from PIL import Image
import os


class ImagePreprocessor:
    """Image preprocessing for CNN training"""
    
    def __init__(self, target_size=(64, 64), normalize=True):
        """
        Initialize preprocessor
        
        Args:
            target_size: (width, height) for resizing images
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            preprocessed_image: numpy array of shape (channels, height, width)
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Change from (height, width, channels) to (channels, height, width)
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Normalize pixel values
            if self.normalize:
                image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory_path, max_images=None):
        """
        Load and preprocess all images from a directory
        
        Args:
            directory_path: Path to directory containing images
            max_images: Maximum number of images to load (None for all)
            
        Returns:
            images: numpy array of preprocessed images
            valid_paths: list of successfully processed image paths
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        # Get all image files
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(directory_path, filename))
        
        # Limit number of images if specified
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        # Process images
        processed_images = []
        valid_paths = []
        
        for image_path in image_paths:
            processed_image = self.load_and_preprocess_image(image_path)
            if processed_image is not None:
                processed_images.append(processed_image)
                valid_paths.append(image_path)
        
        if processed_images:
            return np.array(processed_images), valid_paths
        else:
            return np.array([]), []
    
    def augment_image(self, image):
        """
        Apply data augmentation to an image
        
        Args:
            image: numpy array of shape (channels, height, width)
            
        Returns:
            augmented_image: augmented image
        """
        # Convert back to PIL for easier manipulation
        image_pil = Image.fromarray(
            np.transpose((image * 255).astype(np.uint8), (1, 2, 0))
        )
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        image_pil = image_pil.rotate(angle, fillcolor=(128, 128, 128))
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image_array = np.array(image_pil).astype(np.float32)
        image_array = np.clip(image_array * brightness_factor, 0, 255)
        
        # Convert back to our format
        image_array = np.transpose(image_array, (2, 0, 1))
        
        if self.normalize:
            image_array = image_array / 255.0
        
        return image_array
    
    def create_augmented_batch(self, images, augmentation_factor=2):
        """
        Create augmented versions of a batch of images
        
        Args:
            images: numpy array of images
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            augmented_images: augmented image batch
        """
        augmented_batch = []
        
        for image in images:
            # Add original image
            augmented_batch.append(image)
            
            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                augmented_batch.append(self.augment_image(image))
        
        return np.array(augmented_batch)
    
    def normalize_batch(self, images):
        """
        Normalize a batch of images
        
        Args:
            images: numpy array of images
            
        Returns:
            normalized_images: normalized images
        """
        return images.astype(np.float32) / 255.0 if not self.normalize else images
    
    @staticmethod
    def denormalize_image(image):
        """
        Convert normalized image back to 0-255 range for display
        
        Args:
            image: normalized image array
            
        Returns:
            denormalized_image: image in 0-255 range
        """
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    @staticmethod
    def save_image(image, filepath):
        """
        Save image to file
        
        Args:
            image: image array (channels, height, width)
            filepath: path to save image
        """
        # Convert to PIL format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(np.transpose(image, (1, 2, 0)))
        image_pil.save(filepath)
        print(f"Image saved to {filepath}")
    
    def get_image_statistics(self, images):
        """
        Compute statistics for a batch of images
        
        Args:
            images: numpy array of images
            
        Returns:
            stats: dictionary with mean, std, min, max per channel
        """
        stats = {}
        for channel in range(images.shape[1]):
            channel_data = images[:, channel, :, :].flatten()
            stats[f'channel_{channel}'] = {
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'min': np.min(channel_data),
                'max': np.max(channel_data)
            }
        
        return stats 