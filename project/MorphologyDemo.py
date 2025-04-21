import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from AerialImageLoader import AerialImageLoader

def demonstrate_morphological_operations(image_path):
    """
    Demonstrates various morphological operations on a binary image.
    
    Args:
        image_path: Path to the image file
    """
    # Create image loader and load image
    loader = AerialImageLoader()
    original = loader.load_image(image_path)
    
    if original is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Get image properties
    properties = loader.get_image_properties()
    print("\nImage Properties:")
    for key, value in properties.items():
        print(f"{key}: {value}")
    
    # Convert to grayscale and apply threshold to get binary image for morphology
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create figure with 3x3 subplots
    plt.figure(figsize=(15, 12))
    
    # Display original image
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Display binary image
    plt.subplot(3, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    # 1. Erosion
    kernel_size = 5
    erosion = loader.apply_erosion(kernel_size=kernel_size, iterations=1)
    plt.subplot(3, 3, 3)
    plt.imshow(erosion, cmap='gray')
    plt.title(f"Erosion (kernel={kernel_size})")
    plt.axis('off')
    
    # 2. Dilation
    dilation = loader.apply_dilation(kernel_size=kernel_size, iterations=1)
    plt.subplot(3, 3, 4)
    plt.imshow(dilation, cmap='gray')
    plt.title(f"Dilation (kernel={kernel_size})")
    plt.axis('off')
    
    # 3. Opening (Erosion followed by Dilation)
    opening = loader.apply_opening(kernel_size=kernel_size, iterations=1)
    plt.subplot(3, 3, 5)
    plt.imshow(opening, cmap='gray')
    plt.title(f"Opening (kernel={kernel_size})")
    plt.axis('off')
    
    # 4. Closing (Dilation followed by Erosion)
    closing = loader.apply_closing(kernel_size=kernel_size, iterations=1)
    plt.subplot(3, 3, 6)
    plt.imshow(closing, cmap='gray')
    plt.title(f"Closing (kernel={kernel_size})")
    plt.axis('off')
    
    # 5. Morphological Gradient (Dilation - Erosion)
    gradient = loader.apply_morphological_gradient(kernel_size=kernel_size)
    plt.subplot(3, 3, 7)
    plt.imshow(gradient, cmap='gray')
    plt.title(f"Morphological Gradient (kernel={kernel_size})")
    plt.axis('off')
    
    # 6. Top Hat (Original - Opening)
    top_hat = loader.apply_top_hat(kernel_size=kernel_size)
    plt.subplot(3, 3, 8)
    plt.imshow(top_hat, cmap='gray')
    plt.title(f"Top Hat (kernel={kernel_size})")
    plt.axis('off')
    
    # 7. Black Hat (Closing - Original)
    black_hat = loader.apply_black_hat(kernel_size=kernel_size)
    plt.subplot(3, 3, 9)
    plt.imshow(black_hat, cmap='gray')
    plt.title(f"Black Hat (kernel={kernel_size})")
    plt.axis('off')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.suptitle(f"Morphological Operations with Kernel Size {kernel_size}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Second figure for advanced morphological operations
    plt.figure(figsize=(15, 10))
    
    # 1. Skeletonization
    skeleton = loader.skeletonize_image()
    plt.subplot(2, 2, 1)
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeletonization")
    plt.axis('off')
    
    # 2. Boundary Extraction
    boundaries = loader.extract_boundaries_with_morphology(kernel_size=kernel_size)
    plt.subplot(2, 2, 2)
    plt.imshow(boundaries, cmap='gray')
    plt.title(f"Boundary Extraction (kernel={kernel_size})")
    plt.axis('off')
    
    # 3. Salt and Pepper Noise Removal
    # First, add some salt and pepper noise to the binary image
    noisy_img = binary.copy()
    # Add salt (white) noise
    salt_coords = [np.random.randint(0, i - 1, int(binary.size * 0.02)) for i in binary.shape]
    noisy_img[salt_coords[0], salt_coords[1]] = 255
    # Add pepper (black) noise
    pepper_coords = [np.random.randint(0, i - 1, int(binary.size * 0.02)) for i in binary.shape]
    noisy_img[pepper_coords[0], pepper_coords[1]] = 0
    
    plt.subplot(2, 2, 3)
    plt.imshow(noisy_img, cmap='gray')
    plt.title("Noisy Image (Salt & Pepper)")
    plt.axis('off')
    
    # Save noisy image to loader to apply operations
    loader.image = noisy_img
    
    # Remove noise with morphology
    denoised = loader.remove_noise_with_morphology(kernel_size=3, noise_type='salt_pepper')
    plt.subplot(2, 2, 4)
    plt.imshow(denoised, cmap='gray')
    plt.title("Denoised with Morphology")
    plt.axis('off')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.suptitle("Advanced Morphological Operations", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Third figure for enhanced segmentation
    plt.figure(figsize=(15, 8))
    
    # Load back the original image
    loader.load_image(image_path)
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Basic Otsu segmentation
    otsu = loader.otsu_segmentation()
    plt.subplot(1, 3, 2)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu Segmentation")
    plt.axis('off')
    
    # Enhanced segmentation with morphology
    operations = [
        {'op': 'open', 'kernel_size': 3, 'iterations': 1, 'kernel_shape': 'rect'},
        {'op': 'close', 'kernel_size': 7, 'iterations': 1, 'kernel_shape': 'ellipse'}
    ]
    enhanced = loader.enhance_segmentation_with_morphology(
        segmentation_method='otsu',
        morphology_operations=operations
    )
    
    plt.subplot(1, 3, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Segmentation with Morphology")
    plt.axis('off')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.suptitle("Enhancing Segmentation with Morphological Operations", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    print(f"Selected image: {os.path.basename(image_path)}")
    
    # Demonstrate morphological operations
    demonstrate_morphological_operations(image_path)

if __name__ == "__main__":
    main()