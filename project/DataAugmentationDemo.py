import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from DataAugmentation import DataAugmentation
import matplotlib.gridspec as gridspec

def demonstrate_augmentation_types(image_path):
    """
    Demonstrates different types of augmentation on an image.
    
    Args:
        image_path: Path to the image file
    """
    # Load image
    aug = DataAugmentation()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Demonstrating augmentations on: {os.path.basename(image_path)}")
    print(f"Image shape: {image.shape}")
    
    # Create figure for all augmentation types
    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(6, 4, figure=fig)
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(image_rgb)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    # 1. Brightness variations
    ax2 = fig.add_subplot(gs[0, 2])
    bright_low = aug.augment_brightness(image, brightness_range=(0.6, 0.6))
    ax2.imshow(cv2.cvtColor(bright_low, cv2.COLOR_BGR2RGB))
    ax2.set_title("Brightness (0.6)", fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 3])
    bright_high = aug.augment_brightness(image, brightness_range=(1.3, 1.3))
    ax3.imshow(cv2.cvtColor(bright_high, cv2.COLOR_BGR2RGB))
    ax3.set_title("Brightness (1.3)", fontsize=12)
    ax3.axis('off')
    
    # 2. Contrast variations
    ax4 = fig.add_subplot(gs[1, 0])
    contrast_low = aug.augment_contrast(image, contrast_range=(0.7, 0.7))
    ax4.imshow(cv2.cvtColor(contrast_low, cv2.COLOR_BGR2RGB))
    ax4.set_title("Contrast (0.7)", fontsize=12)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    contrast_high = aug.augment_contrast(image, contrast_range=(1.3, 1.3))
    ax5.imshow(cv2.cvtColor(contrast_high, cv2.COLOR_BGR2RGB))
    ax5.set_title("Contrast (1.3)", fontsize=12)
    ax5.axis('off')
    
    # 3. Noise variations
    ax6 = fig.add_subplot(gs[1, 2])
    gauss_noise = aug.add_gaussian_noise(image, std=20)
    ax6.imshow(cv2.cvtColor(gauss_noise, cv2.COLOR_BGR2RGB))
    ax6.set_title("Gaussian Noise", fontsize=12)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 3])
    sp_noise = aug.add_salt_pepper_noise(image, salt_prob=0.03, pepper_prob=0.03)
    ax7.imshow(cv2.cvtColor(sp_noise, cv2.COLOR_BGR2RGB))
    ax7.set_title("Salt & Pepper Noise", fontsize=12)
    ax7.axis('off')
    
    # 4. Blur variations
    ax8 = fig.add_subplot(gs[2, 0])
    gauss_blur = aug.apply_blur(image, blur_type='gaussian', kernel_size=5)
    ax8.imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))
    ax8.set_title("Gaussian Blur", fontsize=12)
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[2, 1])
    motion_blur = aug.apply_blur(image, blur_type='motion', kernel_size=5)
    ax9.imshow(cv2.cvtColor(motion_blur, cv2.COLOR_BGR2RGB))
    ax9.set_title("Motion Blur", fontsize=12)
    ax9.axis('off')
    
    # 5. Color augmentation
    ax10 = fig.add_subplot(gs[2, 2])
    color_aug = aug.color_augmentation(image)
    ax10.imshow(cv2.cvtColor(color_aug, cv2.COLOR_BGR2RGB))
    ax10.set_title("Color Augmentation", fontsize=12)
    ax10.axis('off')
    
    # 6. Geometric transformations
    ax11 = fig.add_subplot(gs[2, 3])
    albumentations_aug = aug.albumentations_pipeline(image=image)['image']
    ax11.imshow(cv2.cvtColor(albumentations_aug, cv2.COLOR_BGR2RGB))
    ax11.set_title("Geometric Transforms", fontsize=12)
    ax11.axis('off')
    
    # 7. Elastic transformation
    ax12 = fig.add_subplot(gs[3, 0])
    elastic_img = aug.elastic_transform(image, alpha=50)
    ax12.imshow(cv2.cvtColor(elastic_img, cv2.COLOR_BGR2RGB))
    ax12.set_title("Elastic Transform", fontsize=12)
    ax12.axis('off')
    
    # 8. Combined augmentation samples
    for i in range(5):
        ax = fig.add_subplot(gs[4 + (i // 4), i % 4])
        params = {
            'geometry': np.random.choice([True, False]),
            'brightness': np.random.choice([True, False]),
            'contrast': np.random.choice([True, False]),
            'noise': np.random.choice([True, False]),
            'blur': np.random.choice([True, False]),
            'color': np.random.choice([True, False])
        }
        combined = aug.augment_single_image(image, params)
        ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Random Combo {i+1}", fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_augmentation_pipelines(image_path):
    """
    Demonstrates different augmentation pipeline severities.
    
    Args:
        image_path: Path to the image file
    """
    # Load image
    aug = DataAugmentation()
    image = cv2.imread(image_path)
    
    severities = ['light', 'medium', 'heavy']
    
    # Create figure
    fig, axes = plt.subplots(len(severities), 4, figsize=(16, 12))
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i, severity in enumerate(severities):
        # Create pipeline
        pipeline = aug.advanced_augmentation_pipeline(severity)
        
        # Original image
        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f"{severity.capitalize()} Pipeline - Original")
        axes[i, 0].axis('off')
        
        # Apply pipeline multiple times
        for j in range(1, 4):
            augmented = pipeline(image=image)['image']
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(augmented_rgb)
            axes[i, j].set_title(f"Sample {j}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def demonstrate_dataset_augmentation(dataset_dir):
    """
    Demonstrates dataset augmentation with before/after comparison.
    
    Args:
        dataset_dir: Directory containing a small sample dataset
    """
    # Create temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory() as output_dir:
        # Augment dataset
        aug = DataAugmentation()
        
        stats = aug.augment_dataset(
            dataset_dir,
            output_dir,
            augmentation_factor=3,
            augmentation_params={'brightness': True, 'contrast': True, 'noise': True}
        )
        
        print("\nAugmentation Statistics:")
        print(f"Original images: {stats['original_images']}")
        print(f"Augmented images: {stats['augmented_images']}")
        print(f"Failed images: {stats['failed_images']}")
        
        # Visualize before/after for each class
        from pathlib import Path
        
        input_path = Path(dataset_dir)
        output_path = Path(output_dir)
        
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                print(f"\nProcessing class: {class_name}")
                
                # Load original images
                original_images = []
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            original_images.append(img)
                        if len(original_images) >= 2:  # Take first 2 for visualization
                            break
                
                # Load augmented images
                class_output_dir = output_path / class_name
                augmented_images = []
                if class_output_dir.exists():
                    for img_file in class_output_dir.glob('aug_*'):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            augmented_images.append(img)
                        if len(augmented_images) >= 4:  # Take first 4 for visualization
                            break
                
                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()
                
                # Display original images
                for i, img in enumerate(original_images[:2]):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"Original {i+1}")
                    axes[i].axis('off')
                
                # Display augmented images
                for i, img in enumerate(augmented_images[:4]):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i+2].imshow(img_rgb)
                    axes[i+2].set_title(f"Augmented {i+1}")
                    axes[i+2].axis('off')
                
                plt.suptitle(f"Class: {class_name}", fontsize=16)
                plt.tight_layout()
                plt.show()

def demonstrate_training_comparison(dataset_dir):
    """
    Demonstrates impact of augmentation on model training.
    
    Args:
        dataset_dir: Directory containing dataset
    """
    from AerialImageLoader import AerialImageLoader
    
    # Create loader
    loader = AerialImageLoader()
    
    # Train without augmentation
    print("Training without augmentation...")
    try:
        history_no_aug = loader.train_cnn_classifier(
            dataset_dir,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
    except Exception as e:
        print(f"Training without augmentation failed: {e}")
        history_no_aug = None
    
    # Train with augmentation
    print("Training with augmentation...")
    try:
        history_with_aug = loader.train_with_augmentation(
            dataset_dir,
            augmentation_factor=5,
            model_type='CNN',
            epochs=10,
            batch_size=32,
            augmentation_params={'brightness': True, 'contrast': True, 'noise': True, 'blur': True}
        )
    except Exception as e:
        print(f"Training with augmentation failed: {e}")
        history_with_aug = None
    
    # Compare results
    if history_no_aug and history_with_aug:
        # Plot training history comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.plot(history_no_aug.history['accuracy'], label='Training (No Aug)', linestyle='--')
        ax1.plot(history_no_aug.history['val_accuracy'], label='Validation (No Aug)', linestyle='--')
        ax1.plot(history_with_aug.history['accuracy'], label='Training (With Aug)')
        ax1.plot(history_with_aug.history['val_accuracy'], label='Validation (With Aug)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss comparison
        ax2.plot(history_no_aug.history['loss'], label='Training (No Aug)', linestyle='--')
        ax2.plot(history_no_aug.history['val_loss'], label='Validation (No Aug)', linestyle='--')
        ax2.plot(history_with_aug.history['loss'], label='Training (With Aug)')
        ax2.plot(history_with_aug.history['val_loss'], label='Validation (With Aug)')
        ax2.set_title('Model Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nTraining Summary:")
        print("-" * 50)
        print("Without Augmentation:")
        print(f"  Final Training Accuracy: {history_no_aug.history['accuracy'][-1]:.3f}")
        print(f"  Final Validation Accuracy: {history_no_aug.history['val_accuracy'][-1]:.3f}")
        print("\nWith Augmentation:")
        print(f"  Final Training Accuracy: {history_with_aug.history['accuracy'][-1]:.3f}")
        print(f"  Final Validation Accuracy: {history_with_aug.history['val_accuracy'][-1]:.3f}")
        print("-" * 50)

def main():
    """Main demonstration function."""
    print("Data Augmentation Demonstration")
    print("===============================")
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Demo options
    print("\nChoose demonstration type:")
    print("1. Augmentation types on single image")
    print("2. Augmentation pipelines")
    print("3. Dataset augmentation")
    print("4. Training comparison")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        # Single image augmentation demonstration
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if image_path:
            demonstrate_augmentation_types(image_path)
        else:
            print("No image selected.")
    
    elif choice == "2":
        # Pipeline demonstration
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if image_path:
            demonstrate_augmentation_pipelines(image_path)
        else:
            print("No image selected.")
    
    elif choice == "3":
        # Dataset augmentation demonstration
        dataset_dir = filedialog.askdirectory(title="Select Sample Dataset Directory")
        
        if dataset_dir:
            demonstrate_dataset_augmentation(dataset_dir)
        else:
            print("No dataset directory selected.")
    
    elif choice == "4":
        # Training comparison
        dataset_dir = filedialog.askdirectory(title="Select Dataset Directory for Training")
        
        if dataset_dir:
            demonstrate_training_comparison(dataset_dir)
        else:
            print("No dataset directory selected.")
    
    else:
        print("Invalid choice.")
    
    print("\nDemonstration completed.")

if __name__ == "__main__":
    main()