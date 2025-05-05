import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import json
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import tensorflow as tf
from PreTrainedModels import PreTrainedModels
from CNNClassifier import CNNClassifier
from ObjectClassifier import ObjectClassifier

class ObjectRecognitionTesting:
    """
    Class for comprehensive testing of object recognition algorithms.
    Evaluates performance of different models on aerial object detection.
    """
    
    def __init__(self):
        """Initialize testing system"""
        self.test_results = {}
        self.performance_metrics = {}
        self.confusion_matrices = {}
        self.timing_results = {}
    
    def load_test_dataset(self, test_dir):
        """
        Load test dataset for evaluation
        
        Args:
            test_dir: Directory containing test data
            
        Returns:
            test_images, test_labels, class_names
        """
        test_images = []
        test_labels = []
        class_names = []
        
        test_path = Path(test_dir)
        
        for class_dir in test_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                class_idx = len(class_names) - 1
                
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            test_images.append(img)
                            test_labels.append(class_idx)
        
        return test_images, test_labels, class_names
    
    def test_cnn_model(self, model, test_images, test_labels, class_names):
        """
        Test CNN classifier performance
        
        Args:
            model: CNN model
            test_images: Test images
            test_labels: True labels
            class_names: Class names
            
        Returns:
            Test results
        """
        print("Testing CNN Model...")
        start_time = time.time()
        
        predictions = []
        confidences = []
        
        for img in test_images:
            predicted_class, confidence = model.predict(img)
            predictions.append(predicted_class)
            confidences.append(confidence)
        
        test_time = time.time() - start_time
        
        # Convert predictions to class indices
        pred_indices = [class_names.index(pred) for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, pred_indices)
        report = classification_report(test_labels, pred_indices, target_names=class_names)
        conf_matrix = confusion_matrix(test_labels, pred_indices)
        
        results = {
            'model_type': 'CNN',
            'accuracy': accuracy,
            'average_confidence': np.mean(confidences),
            'test_time': test_time,
            'fps': len(test_images) / test_time,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        return results
    
    def test_traditional_ml_model(self, model, test_images, test_labels, class_names):
        """
        Test traditional ML classifier performance
        
        Args:
            model: Traditional ML model
            test_images: Test images
            test_labels: True labels
            class_names: Class names
            
        Returns:
            Test results
        """
        print(f"Testing Traditional ML Model ({model.model_type})...")
        start_time = time.time()
        
        predictions = []
        confidences = []
        
        for img in test_images:
            predicted_class, confidence = model.predict(img)
            predictions.append(predicted_class)
            confidences.append(confidence if confidence is not None else 0.0)
        
        test_time = time.time() - start_time
        
        # Convert predictions to class indices
        pred_indices = [class_names.index(pred) for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, pred_indices)
        report = classification_report(test_labels, pred_indices, target_names=class_names)
        conf_matrix = confusion_matrix(test_labels, pred_indices)
        
        results = {
            'model_type': f'Traditional_ML_{model.model_type}',
            'accuracy': accuracy,
            'average_confidence': np.mean(confidences) if confidences else 0.0,
            'test_time': test_time,
            'fps': len(test_images) / test_time,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        return results
    
    def test_yolo_model(self, yolo_model, test_images, test_labels, class_names):
        """
        Test YOLO object detection performance
        
        Args:
            yolo_model: YOLO model
            test_images: Test images
            test_labels: True labels (for top object)
            class_names: Class names
            
        Returns:
            Test results
        """
        print("Testing YOLO Model...")
        start_time = time.time()
        
        predictions = []
        all_detections = []
        
        for img in test_images:
            # Run YOLO detection
            results = yolo_model.model(img, conf=0.5)
            
            # Get detections
            detections = []
            max_conf = 0
            top_class = None
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    class_name = yolo_model.model.names[int(cls)]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf)
                    })
                    
                    if conf > max_conf:
                        max_conf = conf
                        top_class = class_name
            
            predictions.append(top_class if top_class else 'unknown')
            all_detections.append(detections)
        
        test_time = time.time() - start_time
        
        # Convert predictions to class indices
        pred_indices = []
        for pred in predictions:
            if pred in class_names:
                pred_indices.append(class_names.index(pred))
            else:
                # Assign to first class if not found
                pred_indices.append(0)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, pred_indices)
        report = classification_report(test_labels, pred_indices, target_names=class_names)
        conf_matrix = confusion_matrix(test_labels, pred_indices)
        
        results = {
            'model_type': 'YOLO',
            'accuracy': accuracy,
            'test_time': test_time,
            'fps': len(test_images) / test_time,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'all_detections': all_detections
        }
        
        return results
    
    def test_all_models(self, test_dir, cnn_model=None, ml_models=None, yolo_model=None):
        """
        Test all available models
        
        Args:
            test_dir: Test dataset directory
            cnn_model: CNN classifier
            ml_models: List of traditional ML models
            yolo_model: YOLO detector
            
        Returns:
            Comprehensive test results
        """
        # Load test dataset
        test_images, test_labels, class_names = self.load_test_dataset(test_dir)
        
        print(f"\nTest Dataset Info:")
        print(f"Total images: {len(test_images)}")
        print(f"Classes: {class_names}")
        print(f"Class distribution: {dict(zip(class_names, np.bincount(test_labels)))}")
        print("-" * 50)
        
        # Test CNN model
        if cnn_model is not None:
            cnn_results = self.test_cnn_model(cnn_model, test_images, test_labels, class_names)
            self.test_results['CNN'] = cnn_results
        
        # Test traditional ML models
        if ml_models is not None:
            for ml_model in ml_models:
                ml_results = self.test_traditional_ml_model(ml_model, test_images, test_labels, class_names)
                self.test_results[f'{ml_model.model_type}_ML'] = ml_results
        
        # Test YOLO model
        if yolo_model is not None:
            yolo_results = self.test_yolo_model(yolo_model, test_images, test_labels, class_names)
            self.test_results['YOLO'] = yolo_results
        
        return self.test_results
    
    def generate_performance_report(self, output_dir=None):
        """
        Generate comprehensive performance report
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Performance summary
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Create performance summary
        summary = self._create_performance_summary()
        
        # Generate visualizations
        if output_dir:
            self._generate_confusion_matrices(output_path)
            self._generate_performance_charts(output_path)
            self._generate_timing_analysis(output_path)
            self._generate_detailed_report(output_path, summary)
        
        return summary
    
    def _create_performance_summary(self):
        """Create performance summary table"""
        summary_data = []
        
        for model_name, results in self.test_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'FPS': results['fps'],
                'Test Time (s)': results['test_time'],
                'Avg Confidence': results.get('average_confidence', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def _generate_confusion_matrices(self, output_path):
        """Generate confusion matrix visualizations"""
        n_models = len(self.test_results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 10*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for idx, (model_name, results) in enumerate(self.test_results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_charts(self, output_path):
        """Generate performance comparison charts"""
        # Prepare data
        models = list(self.test_results.keys())
        accuracies = [self.test_results[m]['accuracy'] for m in models]
        fps_values = [self.test_results[m]['fps'] for m in models]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['#2E86C1', '#28B463', '#F39C12', '#E74C3C'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # FPS comparison
        bars2 = ax2.bar(models, fps_values, color=['#3498DB', '#2ECC71', '#F1C40F', '#E67E22'])
        ax2.set_ylabel('FPS (Frames Per Second)')
        ax2.set_title('Model Speed Comparison')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_timing_analysis(self, output_path):
        """Generate timing analysis visualization"""
        models = list(self.test_results.keys())
        test_times = [self.test_results[m]['test_time'] for m in models]
        fps_values = [self.test_results[m]['fps'] for m in models]
        
        # Create pie chart for processing time distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Test time distribution
        ax1.pie(test_times, labels=models, autopct='%1.1f%%', colors=['#3498DB', '#2ECC71', '#F1C40F', '#E67E22'])
        ax1.set_title('Test Time Distribution')
        
        # FPS bar chart
        bars = ax2.bar(models, fps_values, color=['#3498DB', '#2ECC71', '#F1C40F', '#E67E22'])
        ax2.set_ylabel('FPS')
        ax2.set_title('Processing Speed (FPS)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_detailed_report(self, output_path, summary_df):
        """Generate detailed report with all metrics"""
        # Save summary table
        summary_df.to_csv(output_path / 'performance_summary.csv', index=False)
        
        # Create detailed report
        with open(output_path / 'detailed_report.txt', 'w') as f:
            f.write("AERIAL OBJECT RECOGNITION TESTING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            for model_name, results in self.test_results.items():
                f.write(f"\n{model_name} DETAILED RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Accuracy: {results['accuracy']:.3f}\n")
                f.write(f"Test Time: {results['test_time']:.2f} seconds\n")
                f.write(f"FPS: {results['fps']:.2f}\n")
                
                if 'average_confidence' in results:
                    f.write(f"Average Confidence: {results['average_confidence']:.3f}\n")
                
                f.write("\n\nClassification Report:\n")
                f.write(results['classification_report'])
                f.write("\n\n")
    
    def cross_model_analysis(self, test_images, test_labels, class_names):
        """
        Perform cross-model error analysis
        
        Args:
            test_images: Test images
            test_labels: True labels
            class_names: Class names
            
        Returns:
            Error analysis results
        """
        if not self.test_results:
            raise ValueError("No test results available. Run test_all_models first.")
        
        error_analysis = {}
        
        # For each image, check which models got it right/wrong
        for idx, (img, true_label) in enumerate(zip(test_images, test_labels)):
            true_class = class_names[true_label]
            error_analysis[idx] = {
                'true_class': true_class,
                'predictions': {}
            }
            
            for model_name, results in self.test_results.items():
                # Get prediction for this image
                if model_name == 'CNN' or '_ML' in model_name:
                    # For classifiers, get direct predictions
                    if hasattr(results, 'predictions'):
                        pred = results['predictions'][idx]
                    else:
                        # Need to predict again
                        if model_name == 'CNN':
                            pred, _ = cnn_model.predict(img)
                        else:
                            # Get corresponding ML model
                            for ml_model in ml_models:
                                if ml_model.model_type in model_name:
                                    pred, _ = ml_model.predict(img)
                                    break
                elif model_name == 'YOLO':
                    # For YOLO, get top detection
                    if 'all_detections' in results:
                        detections = results['all_detections'][idx]
                        if detections:
                            # Get detection with highest confidence
                            top_det = max(detections, key=lambda x: x['confidence'])
                            pred = top_det['class']
                        else:
                            pred = 'no_detection'
                    else:
                        pred = 'unknown'
                
                error_analysis[idx]['predictions'][model_name] = pred
                error_analysis[idx]['predictions'][f'{model_name}_correct'] = (pred == true_class)
        
        return error_analysis
    
    def generate_error_analysis_report(self, error_analysis, output_path):
        """Generate error analysis visualization"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create agreement matrix
        models = list(self.test_results.keys())
        n_models = len(models)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for img_idx, img_data in error_analysis.items():
            for i, model_i in enumerate(models):
                for j, model_j in enumerate(models):
                    if i < j:  # Upper triangle only
                        pred_i = img_data['predictions'][model_i]
                        pred_j = img_data['predictions'][model_j]
                        if pred_i == pred_j:
                            agreement_matrix[i, j] += 1
                            agreement_matrix[j, i] += 1
        
        # Normalize agreement matrix
        n_images = len(error_analysis)
        agreement_matrix /= n_images
        
        # Visualize agreement matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, xticklabels=models, yticklabels=models, 
                    annot=True, fmt='.2f', cmap='viridis')
        plt.title('Model Agreement Matrix\n(Proportion of Images with Same Predictions)')
        plt.tight_layout()
        plt.savefig(output_path / 'model_agreement_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-class accuracy comparison
        class_accuracies = {}
        for img_idx, img_data in error_analysis.items():
            true_class = img_data['true_class']
            if true_class not in class_accuracies:
                class_accuracies[true_class] = {model: [] for model in models}
            
            for model in models:
                correct = img_data['predictions'][f'{model}_correct']
                class_accuracies[true_class][model].append(1 if correct else 0)
        
        # Calculate average accuracy per class
        class_acc_summary = {}
        for class_name, model_results in class_accuracies.items():
            class_acc_summary[class_name] = {}
            for model, accuracies in model_results.items():
                class_acc_summary[class_name][model] = np.mean(accuracies)
        
        # Visualize per-class accuracy
        df_class_acc = pd.DataFrame(class_acc_summary).T
        
        plt.figure(figsize=(12, 8))
        df_class_acc.plot(kind='bar', width=0.8)
        plt.title('Per-Class Accuracy by Model')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed error analysis
        with open(output_path / 'error_analysis_report.txt', 'w') as f:
            f.write("CROSS-MODEL ERROR ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Agreement Summary:\n")
            f.write("-" * 30 + "\n")
            for i, model_i in enumerate(models):
                for j, model_j in enumerate(models):
                    if i < j:
                        f.write(f"{model_i} vs {model_j}: {agreement_matrix[i,j]:.2%} agreement\n")
            
            f.write("\n\nPer-Class Accuracy Summary:\n")
            f.write("-" * 30 + "\n")
            for class_name, model_accs in class_acc_summary.items():
                f.write(f"\n{class_name}:\n")
                for model, acc in model_accs.items():
                    f.write(f"  {model}: {acc:.2%}\n")
    
    def save_test_results(self, output_path):
        """Save test results to file"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_to_save = {}
        for model_name, results in self.test_results.items():
            results_copy = results.copy()
            # Convert numpy arrays to lists for JSON serialization
            if 'confusion_matrix' in results_copy:
                results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
            results_to_save[model_name] = results_copy
        
        with open(output_path / 'test_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=4)