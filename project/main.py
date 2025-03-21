import numpy as np
import os
import cv2
import glob
from tkinter import Tk, filedialog
from pathlib import Path
import matplotlib.pyplot as plt

class AerialImageLoader:
    """
    Simplified class for loading and displaying aerial images.
    """

    def __init__(self, image_dir=None):
        """
        Initialize the image loader object.

        Args:
            image_dir: Directory with images (optional)
        """
        self.image_dir = image_dir
        self.image_paths = []
        self.image = None
        self.image_path = None

        if image_dir and os.path.exists(image_dir):
            self._load_images_from_directory()

    def _load_images_from_directory(self):
        """Loads paths to all images from the specified directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))

        print(f"Found {len(self.image_paths)} images")

    def load_image(self, image_path):
        """
        Loads an image from the specified path.

        Args:
            image_path: Path to the image

        Returns:
            Loaded image in OpenCV format (NumPy array)
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Load using OpenCV
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image from: {self.image_path}")

        return self.image

    def display_image(self, image, title="Aerial Image"):
        """
        Displays an image using OpenCV.

        Args:
            image: Image to display (NumPy array)
            title: Window title
        """
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_and_display(self, image_path, title=None):
        """
        Loads and displays an image in one operation.

        Args:
            image_path: Path to the image
            title: Window title (optional)
        """
        image = self.load_image(image_path)

        if title is None:
            # Use the file name as the title if not specified
            title = os.path.basename(image_path)

        self.display_image(image, title)

    def get_image_properties(self):
        """
        Returns the properties of the loaded image.

        Returns:
            dict: A dictionary containing image properties.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        height, width, channels = self.image.shape
        return {
            "Path": str(self.image_path),
            "Width": width,
            "Height": height,
            "Channels": channels,
            "Size (bytes)": os.path.getsize(self.image_path)
        }

    def plot_brightness_histogram(self):
        """
        Plots the brightness histogram of the loaded image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        plt.hist(gray_image.ravel(), bins=256, range=[0, 256])
        plt.title('Brightness Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def enhance_contrast(self, method='histogram_equalization'):
        """
        Enhances the contrast of the loaded image.

        Args:
            method (str): The method to use for contrast enhancement ('histogram_equalization' or 'clahe').

        Returns:
            np.ndarray: The contrast-enhanced image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if method == 'histogram_equalization':
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            enhanced_image = cv2.equalizeHist(gray_image)
        elif method == 'clahe':
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)
        else:
            raise ValueError("Unknown enhancement method")

        return enhanced_image
def main():
    """Main function to load and display images."""
    # Create a Tkinter root window (it will not be shown)
    root = Tk()
    root.withdraw()

    # Open a file dialog to select one or multiple images
    image_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])

    if not image_paths:
        print("No images selected")
        return

    # Create an image loader object
    loader = AerialImageLoader()

    # Load and display each selected image
    for image_path in image_paths:
        loader.load_and_display(image_path)

        # Print image properties
        properties = loader.get_image_properties()
        for key, value in properties.items():
            print(f"{key}: {value}")

        # Plot brightness histogram
        loader.plot_brightness_histogram()

        # Enhance contrast and display the enhanced image
        enhanced_image = loader.enhance_contrast(method='clahe')
        cv2.imshow('Enhanced Image', enhanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()