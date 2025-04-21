import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from AerialImageLoader import AerialImageLoader

def demonstrate_geometric_transforms(image_path):
    """
    Demonstrates various geometric transformations on an image.
    
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
    
    # Create a figure for displaying results
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # 1. Resize (scale by 0.5)
    resized = loader.resize_image(scale=0.5)
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title("Resized (0.5x)")
    plt.axis('off')
    
    # 2. Rotation (45 degrees)
    rotated = loader.rotate_image(angle=45)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title("Rotated (45°)")
    plt.axis('off')
    
    # 3. Perspective transform (simulate viewpoint change)
    height, width = original.shape[:2]
    
    # Define source points (rectangle)
    src_points = np.array([
        [0, 0],                  # top-left
        [width - 1, 0],          # top-right
        [width - 1, height - 1], # bottom-right
        [0, height - 1]          # bottom-left
    ], dtype=np.float32)
    
    # Define destination points (trapezoid - simulate perspective)
    offset = int(width * 0.2)  # 20% of width
    dst_points = np.array([
        [offset, 0],             # top-left
        [width - 1 - offset, 0], # top-right
        [width - 1, height - 1], # bottom-right
        [0, height - 1]          # bottom-left
    ], dtype=np.float32)
    
    perspective = loader.perspective_transform(src_points, dst_points)
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    plt.title("Perspective Transform")
    plt.axis('off')
    
    # 4. Crop (center 50%)
    h, w = original.shape[:2]
    x = w // 4
    y = h // 4
    width = w // 2
    height = h // 2
    cropped = loader.crop_image(x, y, width, height)
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped (center)")
    plt.axis('off')
    
    # 5. Flip (horizontal)
    flipped = loader.flip_image(flip_code=1)  # 1 = horizontal flip
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
    plt.title("Flipped (horizontal)")
    plt.axis('off')
    
    # Adjust layout and show
    plt.tight_layout()
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
    
    # Demonstrate geometric transformations
    demonstrate_geometric_transforms(image_path)

if __name__ == "__main__":
    main()