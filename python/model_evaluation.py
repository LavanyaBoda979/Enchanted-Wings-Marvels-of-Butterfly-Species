"""
Comprehensive model evaluation and analysis for butterfly classification
Includes performance metrics, visualization, and error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import json
from collections import defaultdict
import os

class ButterflyModelEvaluator:
    """
    Comprehensive evaluation suite for butterfly classification models
    
    Features:
    - Detailed performance metrics
    - Confusion matrix analysis
    - Per-species performance evaluation
    - Error analysis and visualization
    - Model comparison utilities
    """
    
    def __init__(self, model_path, class_names_path=None):
        self.model = load_model(model_path)
        self.class_names = self._load_class_names(class_names_path, model_path)
        self.num_classes = len(self.class_names)
        
    def _load_class_names(self, class_names_path, model_path):
        """Load class names from file or generate default names"""
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                return json.load(f)
        
        # Try to load from model directory
        default_path = model_path.replace('.h5', '_classes.json')
        if os.path.exists(default_path):
            with open(default_path, 'r') as f:
                return json.load(f)
        
        # Generate default class names
        return [f"Species_{i}" for i in range(self.model.output_shape[-1])]
    
    def evaluate_comprehensive(self, test_generator, save_results=True, output_dir='evaluation_results'):
        """
        Perform comprehensive model evaluation
        
        Args:
            test_generator: Test data generator
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results
        
        Returns:
            dict: Comprehensive evaluation results
        """
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        print("Performing comprehensive model evaluation...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Basic metrics
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=0)
        
        # Classification report
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Per-class analysis
        per_class_metrics = self._calculate_per_class_metrics(
            true_classes, predicted_classes, predictions
        )
        
        # Top-k accuracy
        top_k_accuracies = self._calculate_top_k_accuracy(predictions, true_classes)
        
        # Error analysis
        error_analysis = self._perform_error_analysis(
            true_classes, predicted_classes, predictions, test_generator
        )
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'top_k_accuracies': top_k_accuracies,
            'error_analysis': error_analysis
        }
        
        if save_results:
            self._save_results(results, output_dir)
        
        return results
    
    def _calculate_per_class_metrics(self, true_classes, predicted_classes, predictions):
        """Calculate detailed per-class performance metrics"""
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # Get indices for this class
            class_indices = np.where(true_classes == i)[0]
            
            if len(class_indices) == 0:
                continue
            
            # True positives, false positives, false negatives
            tp = np.sum((true_classes == i) & (predicted_classes == i))
            fp = np.sum((true_classes != i) & (predicted_classes == i))
            fn = np.sum((true_classes == i) & (predicted_classes != i))
            tn = np.sum((true_classes != i) & (predicted_classes != i))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Average confidence for correct predictions
            correct_predictions = class_indices[predicted_classes[class_indices] == i]
            avg_confidence = np.mean(predictions[correct_predictions, i]) if len(correct_predictions) > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': len(class_indices),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'average_confidence': float(avg_confidence)
            }
        
        return per_class_metrics
    
    def _calculate_top_k_accuracy(self, predictions, true_classes, k_values=[1, 3, 5]):
        """Calculate top-k accuracy for different k values"""
        
        top_k_accuracies = {}
        
        for k in k_values:
            if k > self.num_classes:
                continue
            
            # Get top-k predictions for each sample
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
            
            # Check if true class is in top-k predictions
            correct = 0
            for i, true_class in enumerate(true_classes):
                if true_class in top_k_preds[i]:
                    correct += 1
            
            top_k_accuracies[f'top_{k}_accuracy'] = correct / len(true_classes)
        
        return top_k_accuracies
    
    def _perform_error_analysis(self, true_classes, predicted_classes, predictions, test_generator):
        """Perform detailed error analysis"""
        
        # Find misclassified samples
        misclassified_indices = np.where(true_classes != predicted_classes)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(true_classes),
            'most_confused_pairs': [],
            'low_confidence_errors': [],
            'high_confidence_errors': []
        }
        
        # Most confused class pairs
        confusion_pairs = defaultdict(int)
        for idx in misclassified_indices:
            true_class = self.class_names[true_classes[idx]]
            pred_class = self.class_names[predicted_classes[idx]]
            confusion_pairs[(true_class, pred_class)] += 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        error_analysis['most_confused_pairs'] = sorted_pairs[:10]
        
        # Analyze confidence of errors
        for idx in misclassified_indices:
            confidence = predictions[idx, predicted_classes[idx]]
            error_info = {
                'true_class': self.class_names[true_classes[idx]],
                'predicted_class': self.class_names[predicted_classes[idx]],
                'confidence': float(confidence),
                'image_index': int(idx)
            }
            
            if confidence > 0.8:
                error_analysis['high_confidence_errors'].append(error_info)
            elif confidence < 0.5:
                error_analysis['low_confidence_errors'].append(error_info)
        
        # Sort by confidence
        error_analysis['high_confidence_errors'].sort(key=lambda x: x['confidence'], reverse=True)
        error_analysis['low_confidence_errors'].sort(key=lambda x: x['confidence'])
        
        return error_analysis
    
    def plot_confusion_matrix(self, cm, normalize=True, figsize=(15, 12), save_path=None):
        """Plot detailed confusion matrix"""
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm_norm = cm
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_performance(self, per_class_metrics, save_path=None):
        """Plot per-class performance metrics"""
        
        # Extract metrics for plotting
        species_names = list(per_class_metrics.keys())
        precisions = [per_class_metrics[species]['precision'] for species in species_names]
        recalls = [per_class_metrics[species]['recall'] for species in species_names]
        f1_scores = [per_class_metrics[species]['f1_score'] for species in species_names]
        supports = [per_class_metrics[species]['support'] for species in species_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Precision by species
        axes[0, 0].barh(range(len(species_names)), precisions, color='skyblue')
        axes[0, 0].set_yticks(range(len(species_names)))
        axes[0, 0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in species_names], fontsize=8)
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_title('Precision by Species')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Recall by species
        axes[0, 1].barh(range(len(species_names)), recalls, color='lightcoral')
        axes[0, 1].set_yticks(range(len(species_names)))
        axes[0, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in species_names], fontsize=8)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_title('Recall by Species')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # F1-Score by species
        axes[1, 0].barh(range(len(species_names)), f1_scores, color='lightgreen')
        axes[1, 0].set_yticks(range(len(species_names)))
        axes[1, 0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in species_names], fontsize=8)
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_title('F1-Score by Species')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Support (number of test samples) by species
        axes[1, 1].barh(range(len(species_names)), supports, color='gold')
        axes[1, 1].set_yticks(range(len(species_names)))
        axes[1, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in species_names], fontsize=8)
        axes[1, 1].set_xlabel('Number of Test Samples')
        axes[1, 1].set_title('Test Sample Distribution')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_analysis(self, error_analysis, save_path=None):
        """Plot error analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Most confused pairs
        if error_analysis['most_confused_pairs']:
            pairs = [f"{pair[0][:15]}→{pair[1][:15]}" for pair, count in error_analysis['most_confused_pairs'][:10]]
            counts = [count for pair, count in error_analysis['most_confused_pairs'][:10]]
            
            axes[0, 0].barh(range(len(pairs)), counts, color='salmon')
            axes[0, 0].set_yticks(range(len(pairs)))
            axes[0, 0].set_yticklabels(pairs, fontsize=10)
            axes[0, 0].set_xlabel('Number of Confusions')
            axes[0, 0].set_title('Most Confused Species Pairs')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # High confidence errors
        if error_analysis['high_confidence_errors']:
            high_conf_errors = error_analysis['high_confidence_errors'][:10]
            error_labels = [f"{err['true_class'][:10]}→{err['predicted_class'][:10]}" 
                           for err in high_conf_errors]
            confidences = [err['confidence'] for err in high_conf_errors]
            
            axes[0, 1].barh(range(len(error_labels)), confidences, color='orange')
            axes[0, 1].set_yticks(range(len(error_labels)))
            axes[0, 1].set_yticklabels(error_labels, fontsize=10)
            axes[0, 1].set_xlabel('Prediction Confidence')
            axes[0, 1].set_title('High Confidence Errors')
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Error distribution by confidence
        all_errors = error_analysis['high_confidence_errors'] + error_analysis['low_confidence_errors']
        if all_errors:
            confidences = [err['confidence'] for err in all_errors]
            axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='purple')
            axes[1, 0].set_xlabel('Prediction Confidence')
            axes[1, 0].set_ylabel('Number of Errors')
            axes[1, 0].set_title('Error Distribution by Confidence')
            axes[1, 0].grid(alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Error Analysis Summary:
        
        Total Errors: {error_analysis['total_errors']:,}
        Error Rate: {error_analysis['error_rate']:.3f} ({error_analysis['error_rate']*100:.1f}%)
        
        High Confidence Errors: {len(error_analysis['high_confidence_errors'])}
        Low Confidence Errors: {len(error_analysis['low_confidence_errors'])}
        
        Most Problematic Pairs:
        """
        
        for i, (pair, count) in enumerate(error_analysis['most_confused_pairs'][:5]):
            summary_text += f"\n{i+1}. {pair[0]} → {pair[1]} ({count} errors)"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _save_results(self, results, output_dir):
        """Save evaluation results to files"""
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(results['classification_report'], f, indent=2)
        
        # Save per-class metrics
        with open(os.path.join(output_dir, 'per_class_metrics.json'), 'w') as f:
            json.dump(results['per_class_metrics'], f, indent=2)
        
        # Save error analysis
        with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
            json.dump(results['error_analysis'], f, indent=2, default=str)
        
        # Save confusion matrix
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), results['confusion_matrix'])
        
        # Save summary
        summary = {
            'test_accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'top_k_accuracies': results['top_k_accuracies'],
            'total_classes': self.num_classes,
            'evaluation_date': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}")

# Example usage
def main():
    """
    Example evaluation pipeline
    """
    
    # Initialize evaluator
    evaluator = ButterflyModelEvaluator(
        model_path='butterfly_classifier_final.h5',
        class_names_path='butterfly_classes.json'
    )
    
    # Load test data (assuming you have a test generator)
    # test_generator = ...
    
    # Perform comprehensive evaluation
    # results = evaluator.evaluate_comprehensive(test_generator)
    
    # Generate visualizations
    # evaluator.plot_confusion_matrix(results['confusion_matrix'])
    # evaluator.plot_per_class_performance(results['per_class_metrics'])
    # evaluator.plot_error_analysis(results['error_analysis'])
    
    print("Model evaluation completed!")

if __name__ == "__main__":
    main()