"""
Visualization utilities for training and results
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_accs: Training accuracies over epochs
        val_accs: Validation accuracies over epochs
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(model, images, labels, num_samples=8, save_path=None):
    """
    Plot sample predictions with images
    
    Args:
        model: Trained CNN model
        images: Sample images
        labels: True labels
        num_samples: Number of samples to show
        save_path: Path to save plot (optional)
    """
    # Get random samples
    indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Get predictions
    predictions, probabilities = model.predict(sample_images)
    
    # Create subplot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['Cat', 'Dog']
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Convert image for display (channels first to channels last)
        img = np.transpose(sample_images[i], (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Create title with prediction info
        true_label = int(sample_labels[i])
        pred_label = int(predictions[i])
        confidence = float(probabilities[i])
        
        if pred_label == 1:
            confidence = confidence
        else:
            confidence = 1 - confidence
        
        title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence:.2f})'
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(title, color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show()


def plot_filters(model, layer_idx=0, num_filters=16, save_path=None):
    """
    Visualize learned filters from a convolutional layer
    
    Args:
        model: Trained CNN model
        layer_idx: Index of convolutional layer to visualize
        num_filters: Number of filters to show
        save_path: Path to save plot (optional)
    """
    # Find convolutional layers
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and len(layer.weights.shape) == 4:
            conv_layers.append((i, layer))
    
    if layer_idx >= len(conv_layers):
        print(f"Layer index {layer_idx} out of range. Available conv layers: {len(conv_layers)}")
        return
    
    layer = conv_layers[layer_idx][1]
    filters = layer.weights[:num_filters]  # Shape: (num_filters, channels, height, width)
    
    # Create subplot grid
    cols = 4
    rows = (num_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get filter and normalize for display
        filter_weights = filters[i]
        
        # If filter has multiple channels, take the mean
        if filter_weights.shape[0] > 1:
            filter_display = np.mean(filter_weights, axis=0)
        else:
            filter_display = filter_weights[0]
        
        # Normalize to [0, 1]
        filter_display = (filter_display - filter_display.min()) / (filter_display.max() - filter_display.min())
        
        ax.imshow(filter_display, cmap='viridis')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Learned Filters - Convolutional Layer {layer_idx + 1}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")
    
    plt.show()


def save_plots(figures_dir='plots'):
    """Create directory for saving plots"""
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created directory: {figures_dir}")
    return figures_dir


def plot_loss_landscape_2d(model, data_loader, save_path=None):
    """
    Plot a simple 2D loss landscape visualization
    
    Args:
        model: Trained CNN model
        data_loader: Data loader for loss calculation
        save_path: Path to save plot (optional)
    """
    print("2D loss landscape visualization not implemented") 