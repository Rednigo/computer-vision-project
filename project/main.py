import cv2
import os
import glob
from tkinter import Tk, filedialog
from pathlib import Path


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
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")

        return image

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


if __name__ == "__main__":
    main()