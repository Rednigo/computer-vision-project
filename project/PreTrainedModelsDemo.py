import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from PreTrainedModels import PreTrainedModels
import time

def demonstrate_pretrained_models(image_path):
    """
    Demonstrates various pre-trained models on an image.
    
    Args:
        image_path: Path to the image file
    """
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Initialize pre-trained models
    models = PreTrainedModels()
    
    # Load the original image
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(original_img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # 1. ResNet50 Classification
    print("\n1. Running ResNet50 classification...")
    start_time = time.time()
    resnet_results = models.predict_resnet(image_path)
    resnet_time = time.time() - start_time
    
    resnet_text = "ResNet50 Predictions:\n"
    for i, (_, description, score) in enumerate(resnet_results[:3]):
        resnet_text += f"{i+1}. {description}\n  Confidence: {score:.3f}\n"
    resnet_text += f"\nProcessing time: {resnet_time:.2f}s"
    
    plt.subplot(3, 3, 2)
    plt.imshow(original_img_rgb)
    plt.title("ResNet50 Results")
    plt.text(0.02, 0.98, resnet_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    plt.axis('off')
    
    # 2. MobileNetV2 Classification
    print("\n2. Running MobileNetV2 classification...")
    start_time = time.time()
    mobilenet_results = models.predict_mobilenet(image_path)
    mobilenet_time = time.time() - start_time
    
    mobilenet_text = "MobileNetV2 Predictions:\n"
    for i, (_, description, score) in enumerate(mobilenet_results[:3]):
        mobilenet_text += f"{i+1}. {description}\n  Confidence: {score:.3f}\n"
    mobilenet_text += f"\nProcessing time: {mobilenet_time:.2f}s"
    
    plt.subplot(3, 3, 3)
    plt.imshow(original_img_rgb)
    plt.title("MobileNetV2 Results")
    plt.text(0.02, 0.98, mobilenet_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    plt.axis('off')
    
    # 3. YOLO Object Detection
    print("\n3. Running YOLO object detection...")
    start_time = time.time()
    yolo_detections, yolo_img = models.detect_yolo(image_path, conf_threshold=0.5)
    yolo_time = time.time() - start_time
    
    yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(3, 3, 4)
    plt.imshow(yolo_img_rgb)
    plt.title("YOLO Detection Results")
    plt.axis('off')
    
    # YOLO statistics
    yolo_text = f"YOLO Statistics:\n\n"
    yolo_text += f"Objects detected: {len(yolo_detections)}\n"
    yolo_text += f"Processing time: {yolo_time:.2f}s\n\n"
    
    # Count classes
    class_counts = {}
    for det in yolo_detections:
        class_name = det['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    yolo_text += "Detected objects:\n"
    for class_name, count in class_counts.items():
        yolo_text += f"- {class_name}: {count}\n"
    
    plt.subplot(3, 3, 5)
    plt.text(0.5, 0.5, yolo_text, transform=plt.gca().transAxes,
             ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    plt.title("YOLO Statistics")
    plt.axis('off')
    
    # 4. U-Net Segmentation
    print("\n4. Running U-Net segmentation...")
    start_time = time.time()
    unet_mask = models.segment_unet(image_path)
    unet_time = time.time() - start_time
    
    # Create visualization of segmentation
    img_resized = cv2.resize(original_img, (256, 256))
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Create colored mask
    colored_mask = np.zeros_like(img_resized_rgb)
    colored_mask[unet_mask > 0] = [0, 255, 0]  # Green for mask
    
    # Overlay mask on image
    overlaid = cv2.addWeighted(img_resized_rgb, 0.7, colored_mask, 0.3, 0)
    
    plt.subplot(3, 3, 6)
    plt.imshow(unet_mask, cmap='gray')
    plt.title("U-Net Segmentation Mask")
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(overlaid)
    plt.title("U-Net Overlay")
    plt.axis('off')
    
    # Segmentation statistics
    unet_text = f"U-Net Statistics:\n\n"
    unet_text += f"Segmented pixels: {np.sum(unet_mask > 0)}\n"
    unet_text += f"Total pixels: {unet_mask.size}\n"
    unet_text += f"Segmented ratio: {np.sum(unet_mask > 0) / unet_mask.size:.2%}\n"
    unet_text += f"Processing time: {unet_time:.2f}s"
    
    plt.subplot(3, 3, 8)
    plt.text(0.5, 0.5, unet_text, transform=plt.gca().transAxes,
             ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    plt.title("U-Net Statistics")
    plt.axis('off')
    
    # Performance comparison
    plt.subplot(3, 3, 9)
    model_names = ['ResNet50', 'MobileNet', 'YOLO', 'U-Net']
    processing_times = [resnet_time, mobilenet_time, yolo_time, unet_time]
    
    bars = plt.bar(model_names, processing_times, color=['blue', 'orange', 'green', 'red'])
    plt.title("Model Processing Times")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create a summary visualization
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original vs YOLO
    axes[0, 0].imshow(original_img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(yolo_img_rgb)
    axes[0, 1].set_title("YOLO Object Detection")
    axes[0, 1].axis('off')
    
    # Segmentation results
    axes[1, 0].imshow(unet_mask, cmap='gray')
    axes[1, 0].set_title("U-Net Segmentation")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlaid)
    axes[1, 1].set_title("Segmentation Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nOVERALL SUMMARY:")
    print("=" * 50)
    print(f"Image processed: {os.path.basename(image_path)}")
    print(f"Total processing time: {resnet_time + mobilenet_time + yolo_time + unet_time:.2f}s")
    print(f"\nTop predictions:")
    print(f"ResNet50: {resnet_results[0][1]} ({resnet_results[0][2]:.3f})")
    print(f"MobileNet: {mobilenet_results[0][1]} ({mobilenet_results[0][2]:.3f})")
    print(f"YOLO: Detected {len(yolo_detections)} objects")
    print(f"U-Net: Segmented {np.sum(unet_mask > 0) / unet_mask.size:.1%} of image")
    print("=" * 50)

def main():
    """Main demonstration function."""
    print("Pre-trained Models Demonstration")
    print("================================")
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to select an image
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    print(f"Selected image: {os.path.basename(image_path)}")
    
    try:
        # Run demonstration
        demonstrate_pretrained_models(image_path)
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDemonstration completed.")

if __name__ == "__main__":
    main()