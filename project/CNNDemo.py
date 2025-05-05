import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from AerialImageLoader import AerialImageLoader

def prepare_sample_data(dataset_dir):
    """
    Prepare and display sample data from the dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    loader = AerialImageLoader()
    
    # Create dataset from directory
    from pathlib import Path
    dataset_path = Path(dataset_dir)
    
    # Collect sample images from each class
    sample_images = {}
    sample_counts = {}
    
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            image_files = []
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                image_files.extend(list(class_dir.glob(ext)))
                
            if image_files:
                # Load first 5 images from each class
                images_in_class = []
                for i, image_file in enumerate(image_files[:5]):
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        # Resize for display
                        resized_img = cv2.resize(image, (128, 128))
                        images_in_class.append(resized_img)
                        
                sample_images[class_name] = images_in_class
                sample_counts[class_name] = len(image_files)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot dataset statistics
    if sample_counts:
        plt.subplot(2, 1, 1)
        classes = list(sample_counts.keys())
        counts = list(sample_counts.values())
        bars = plt.bar(classes, counts)
        plt.title('Dataset Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')
        
        # Plot sample images
        subplot_idx = 0
        for class_name, images in sample_images.items():
            for img in images:
                subplot_idx += 1
                if subplot_idx > 25:  # Limit to 25 images total
                    break
                
                plt.subplot(5, 5, subplot_idx + 5)  # Skip first 5 subplots for the bar chart
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.title(class_name[:10], fontsize=8)
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_cnn_training(dataset_dir):
    """
    Demonstrates CNN training and evaluation.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    print("Initializing AerialImageLoader with CNN support...")
    loader = AerialImageLoader()
    
    try:
        print("\nTraining CNN classifier...")
        print("This may take a while depending on your dataset size and hardware.")
        
        # Train CNN
        history = loader.train_cnn_classifier(
            dataset_dir=dataset_dir,
            epochs=30,  # Use fewer epochs for demo
            batch_size=32,
            validation_split=0.2
        )
        
        print("\nTraining completed!")
        
        # Display training history
        loader.plot_cnn_training_history()
        
        # Get model summary
        print("\nCNN Model Architecture:")
        print(loader.get_cnn_summary())
        
        # Save model
        model_save_path = os.path.join(os.path.dirname(dataset_dir), "cnn_model")
        loader.save_cnn_model(model_save_path)
        print(f"\nModel saved to: {model_save_path}")
        
        return loader
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def demonstrate_cnn_prediction(loader, test_image_path):
    """
    Demonstrates CNN prediction on a single image.
    
    Args:
        loader: Trained AerialImageLoader instance
        test_image_path: Path to test image
    """
    # Load test image
    test_image = loader.load_image(test_image_path)
    
    # Make prediction
    predicted_class, confidence = loader.classify_image_cnn()
    
    # Visualize result
    plt.figure(figsize=(10, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    
    # Prediction result
    plt.subplot(1, 2, 2)
    result_image = loader.draw_classification_result(predicted_class, confidence)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

def compare_classifiers(loader, test_image_path):
    """
    Compare traditional ML classifier vs CNN performance.
    
    Args:
        loader: Trained AerialImageLoader instance
        test_image_path: Path to test image
    """
    # Load test image
    test_image = loader.load_image(test_image_path)
    
    try:
        # CNN prediction
        cnn_pred, cnn_conf = loader.classify_image_cnn()
        
        # Traditional classifier prediction (if available)
        if loader.classifier.model is not None:
            ml_pred, ml_conf = loader.classify_image()
        else:
            ml_pred, ml_conf = "Not trained", 0.0
        
        # Create comparison visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis('off')
        
        # CNN result
        plt.subplot(1, 3, 2)
        cnn_result = loader.draw_classification_result(cnn_pred, cnn_conf)
        plt.imshow(cv2.cvtColor(cnn_result, cv2.COLOR_BGR2RGB))
        plt.title(f"CNN: {cnn_pred[:10]} ({cnn_conf:.2f})")
        plt.axis('off')
        
        # Traditional ML result
        plt.subplot(1, 3, 3)
        ml_result = loader.draw_classification_result(ml_pred, ml_conf)
        plt.imshow(cv2.cvtColor(ml_result, cv2.COLOR_BGR2RGB))
        plt.title(f"Traditional ML: {ml_pred[:10]} ({ml_conf:.2f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Comparison:")
        print(f"CNN Prediction: {cnn_pred} (Confidence: {cnn_conf:.2f})")
        print(f"Traditional ML Prediction: {ml_pred} (Confidence: {ml_conf:.2f})")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")

def main():
    """Main demonstration function."""
    print("CNN Classifier Demonstration")
    print("===========================")
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to select dataset directory
    dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
    
    if not dataset_dir:
        print("No dataset selected. Exiting.")
        return
    
    print(f"Selected dataset: {dataset_dir}")
    
    # Display dataset information
    prepare_sample_data(dataset_dir)
    
    # Train CNN
    loader = demonstrate_cnn_training(dataset_dir)
    
    if loader is None:
        print("Training failed. Cannot proceed with predictions.")
        return
    
    # Ask user to select a test image
    test_image_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    
    if test_image_path:
        print(f"\nTesting with image: {os.path.basename(test_image_path)}")
        
        # Make single prediction
        demonstrate_cnn_prediction(loader, test_image_path)
        
        # Compare with traditional classifier
        compare_classifiers(loader, test_image_path)
    else:
        print("No test image selected. Skipping prediction demonstration.")
    
    print("\nCNN demonstration completed.")

if __name__ == "__main__":
    main()