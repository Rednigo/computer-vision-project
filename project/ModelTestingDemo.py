import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from ObjectRecognitionTesting import ObjectRecognitionTesting
from AerialImageLoader import AerialImageLoader
import json
from datetime import datetime

def run_comprehensive_testing_demo():
    """Run a comprehensive testing demonstration"""
    print("Aerial Object Recognition Testing Demonstration")
    print("=" * 50)
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Get test dataset directory
    print("\nStep 1: Select test dataset directory")
    test_dataset_dir = filedialog.askdirectory(title="Select Test Dataset Directory")
    
    if not test_dataset_dir:
        print("No dataset selected. Exiting.")
        return
    
    # Get output directory
    print("\nStep 2: Select output directory for results")
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    
    if not output_dir:
        print("No output directory selected. Exiting.")
        return
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"test_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTest dataset: {test_dataset_dir}")
    print(f"Output directory: {output_dir}")
    print("\n" + "=" * 50)
    
    # Initialize testing system
    print("\nInitializing testing system...")
    testing_system = ObjectRecognitionTesting()
    loader = AerialImageLoader()
    
    # Load test dataset
    print("\nStep 3: Loading test dataset...")
    test_images, test_labels, class_names = testing_system.load_test_dataset(test_dataset_dir)
    
    print(f"\nTest Dataset Information:")
    print(f"Total images: {len(test_images)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:")
    for i, class_name in enumerate(class_names):
        count = test_labels.count(i)
        print(f"  {class_name}: {count} images")
    
    # Prepare models for testing
    print("\nStep 4: Preparing models for testing...")
    
    # Prepare CNN model
    print("- Preparing CNN model...")
    try:
        # You might need to load a pre-trained model here
        # This is a placeholder - replace with actual model loading
        cnn_model = loader.cnn_classifier
        if hasattr(cnn_model, 'model') and cnn_model.model is None:
            print("  Warning: CNN model not loaded. Skipping CNN testing.")
            cnn_model = None
    except Exception as e:
        print(f"  Error with CNN model: {e}")
        cnn_model = None
    
    # Prepare traditional ML models
    print("- Preparing traditional ML models...")
    ml_models = []
    try:
        # Create SVM model for testing
        from ObjectClassifier import ObjectClassifier
        svm_classifier = ObjectClassifier()
        if loader.classifier.model is not None:
            svm_classifier.model = loader.classifier.model
            svm_classifier.scaler = loader.classifier.scaler
            svm_classifier.classes = loader.classifier.classes
            ml_models.append(svm_classifier)
    except Exception as e:
        print(f"  Error with ML models: {e}")
    
    # Prepare YOLO model
    print("- Preparing YOLO model...")
    try:
        loader.load_yolo()
        yolo_model = loader.pretrained_models
    except Exception as e:
        print(f"  Error with YOLO model: {e}")
        yolo_model = None
    
    # Run testing
    print("\nStep 5: Running comprehensive tests...")
    print("-" * 50)
    
    results = testing_system.test_all_models(
        test_dataset_dir,
        cnn_model=cnn_model,
        ml_models=ml_models,
        yolo_model=yolo_model
    )
    
    # Generate performance report
    print("\nStep 6: Generating performance report...")
    summary_df = testing_system.generate_performance_report(output_dir)
    
    # Print summary to console
    print("\nPerformance Summary:")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    print("-" * 50)
    
    # Perform error analysis
    print("\nStep 7: Performing error analysis...")
    error_analysis = testing_system.cross_model_analysis(test_images, test_labels, class_names)
    testing_system.generate_error_analysis_report(error_analysis, output_dir)
    
    # Generate efficiency comparison
    print("\nStep 8: Generating efficiency comparison...")
    efficiency_metrics = loader.compare_model_efficiency(test_dataset_dir, output_dir)
    
    # Print efficiency metrics
    print("\nEfficiency Metrics:")
    print("-" * 50)
    for model_name, metrics in efficiency_metrics.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  FPS: {metrics['fps']:.2f}")
        print(f"  Efficiency Score: {metrics['accuracy_per_second']:.3f}")
    print("-" * 50)
    
    # Save detailed results
    print("\nStep 9: Saving detailed results...")
    testing_system.save_test_results(output_dir)
    
    # Generate HTML report
    print("\nStep 10: Generating HTML report...")
    generate_html_report(output_dir, results, efficiency_metrics)
    
    print("\n" + "=" * 50)
    print(f"Testing completed successfully!")
    print(f"All results saved to: {output_dir}")
    print("\nFiles generated:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    # Ask if user wants to view report
    if messagebox.askyesno("Testing Complete", 
                          f"Testing completed successfully!\n\nResults saved to:\n{output_dir}\n\nOpen HTML report now?"):
        import webbrowser
        report_path = os.path.join(output_dir, "test_report.html")
        webbrowser.open(f'file://{os.path.abspath(report_path)}')

def generate_html_report(output_dir, results, efficiency_metrics):
    """Generate comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Aerial Object Recognition Testing Report</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                line-height: 1.6;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #333; 
                text-align: center;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            h2 {{ 
                color: #444; 
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            .section {{ 
                margin-bottom: 40px; 
                background-color: #fafafa;
                padding: 20px;
                border-radius: 5px;
            }}
            img {{ 
                max-width: 100%; 
                height: auto; 
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0;
                background-color: white;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #4CAF50; 
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .metric-card {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                display: inline-block;
                min-width: 200px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .timestamp {{
                text-align: center;
                color: #666;
                font-size: 0.9em;
                margin-bottom: 20px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Aerial Object Recognition Testing Report</h1>
            <div class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
    """
    
    # Add summary metrics
    for model_name, model_results in results.items():
        html_content += f"""
                    <div class="metric-card">
                        <h3>{model_name}</h3>
                        <div class="metric-value">{model_results['accuracy']:.1%}</div>
                        <div>Accuracy</div>
                        <div style="margin-top: 10px;">{model_results['fps']:.1f} FPS</div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                <img src="performance_comparison.png" alt="Performance Comparison">
                <p>Comparison of accuracy and processing speed across different models.</p>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>FPS</th>
                        <th>Test Time (s)</th>
                        <th>Avg Confidence</th>
                    </tr>
    """
    
    # Add detailed results table
    for model_name, model_results in results.items():
        confidence = model_results.get('average_confidence', 'N/A')
        if confidence != 'N/A':
            confidence = f"{confidence:.3f}"
        
        html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{model_results['accuracy']:.3f}</td>
                        <td>{model_results['fps']:.2f}</td>
                        <td>{model_results['test_time']:.2f}</td>
                        <td>{confidence}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Confusion Matrices</h2>
                <img src="confusion_matrices.png" alt="Confusion Matrices">
                <p>Visualization of prediction errors for each model.</p>
            </div>
            
            <div class="section">
                <h2>Timing Analysis</h2>
                <img src="timing_analysis.png" alt="Timing Analysis">
                <p>Analysis of processing time distribution and speed metrics.</p>
            </div>
            
            <div class="section">
                <h2>Error Analysis</h2>
                <h3>Model Agreement Matrix</h3>
                <img src="model_agreement_matrix.png" alt="Model Agreement Matrix">
                <p>Shows the proportion of images where models made the same predictions.</p>
                
                <h3>Per-Class Accuracy</h3>
                <img src="per_class_accuracy.png" alt="Per-Class Accuracy">
                <p>Breakdown of model performance by object class.</p>
            </div>
            
            <div class="section">
                <h2>Efficiency Analysis</h2>
                <img src="efficiency_comparison.png" alt="Efficiency Comparison">
                <p>Comparison of accuracy vs speed efficiency. Bubble size represents efficiency score.</p>
            </div>
            
            <div class="section">
                <h2>Conclusions</h2>
                <p>Based on the comprehensive testing results:</p>
                <ul>
                    <li>CNN models generally achieve the highest accuracy but may be slower</li>
                    <li>YOLO provides real-time performance with competitive accuracy</li>
                    <li>Traditional ML models offer good speed-accuracy tradeoffs</li>
                    <li>Model choice depends on specific requirements (accuracy vs speed)</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, "test_report.html"), "w") as f:
        f.write(html_content)

def main():
    """Main function"""
    try:
        run_comprehensive_testing_demo()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()