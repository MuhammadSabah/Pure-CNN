"""
Evaluation metrics for classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy
    
    Args:
        predictions: Model predictions
        targets: True labels
        
    Returns:
        accuracy: Accuracy score
    """
    return np.mean(predictions == targets)


def calculate_precision_recall(predictions, targets):
    """
    Calculate precision and recall for binary classification
    
    Args:
        predictions: Model predictions (0 or 1)
        targets: True labels (0 or 1)
        
    Returns:
        precision, recall, f1_score: Metrics
    """
    # True positives, false positives, false negatives
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def plot_confusion_matrix(predictions, targets, class_names=['Cat', 'Dog'], save_path=None):
    """
    Plot confusion matrix
    
    Args:
        predictions: Model predictions
        targets: True labels
        class_names: Names of classes
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def calculate_auc_roc(probabilities, targets):
    """
    Calculate AUC-ROC for binary classification
    
    Args:
        probabilities: Model prediction probabilities
        targets: True labels
        
    Returns:
        auc: Area under ROC curve
    """
    # Simple AUC calculation
    sorted_indices = np.argsort(probabilities.flatten())
    sorted_targets = targets.flatten()[sorted_indices]
    
    # Calculate TPR and FPR at different thresholds
    tpr_list = []
    fpr_list = []
    
    thresholds = np.unique(probabilities)
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp = np.sum((pred == 1) & (targets == 1))
        fp = np.sum((pred == 1) & (targets == 0))
        tn = np.sum((pred == 0) & (targets == 0))
        fn = np.sum((pred == 0) & (targets == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)
    
    # Sort by FPR
    sorted_indices = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_indices]
    tpr_sorted = tpr_array[sorted_indices]
    
    auc = np.trapz(tpr_sorted, fpr_sorted)
    return auc


def evaluate_model(model, images, labels):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained CNN model
        images: Test images
        labels: True labels
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Get predictions
    predictions, probabilities = model.predict(images)
    
    # Flatten for easier computation
    predictions = predictions.flatten()
    probabilities = probabilities.flatten()
    labels = labels.flatten()
    
    # Calculate metrics
    accuracy = calculate_accuracy(predictions, labels)
    precision, recall, f1_score = calculate_precision_recall(predictions, labels)
    auc = calculate_auc_roc(probabilities, labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auc': auc
    }
    
    return metrics 