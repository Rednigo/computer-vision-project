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

    def denoise_image(self, method='gaussian', **kwargs):
        """
        Denoises the loaded image using the specified method.

        Args:
            method (str): The denoising method to use ('gaussian', 'median', 'bilateral').
            kwargs: Additional parameters for the denoising methods.

        Returns:
            np.ndarray: The denoised image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if method == 'gaussian':
            ksize = kwargs.get('ksize', (5, 5))
            sigma = kwargs.get('sigma', 0)
            denoised_image = cv2.GaussianBlur(self.image, ksize, sigma)
        elif method == 'median':
            ksize = kwargs.get('ksize', 5)
            denoised_image = cv2.medianBlur(self.image, ksize)
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised_image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        else:
            raise ValueError("Unknown denoising method")

        return denoised_image

    def sharpen_image(self, method='unsharp_mask', **kwargs):
        """
        Sharpens the loaded image using the specified method.

        Args:
            method (str): The sharpening method to use ('unsharp_mask', 'laplacian').
            kwargs: Additional parameters for the sharpening methods.

        Returns:
            np.ndarray: The sharpened image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if method == 'unsharp_mask':
            amount = kwargs.get('amount', 1.0)
            blurred = cv2.GaussianBlur(self.image, (0, 0), 3)
            sharpened_image = cv2.addWeighted(self.image, 1 + amount, blurred, -amount, 0)
        elif method == 'laplacian':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_image = cv2.filter2D(self.image, -1, kernel)
        else:
            raise ValueError("Unknown sharpening method")

        return sharpened_image

    def threshold_segmentation(self, threshold_value=127, max_value=255):
        """
        Applies simple threshold segmentation to the loaded image.

        Args:
            threshold_value (int): The threshold value.
            max_value (int): The maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.

        Returns:
            np.ndarray: The thresholded image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
        return thresholded_image

    def otsu_segmentation(self):
        """
        Applies Otsu's thresholding method to the loaded image.

        Returns:
            np.ndarray: The thresholded image using Otsu's method.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_image

    def watershed_segmentation(self):
        """
        Applies the Watershed algorithm to the loaded image.

        Returns:
            np.ndarray: The segmented image using the Watershed algorithm.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(self.image, markers)
        self.image[markers == -1] = [255, 0, 0]

        return self.image

    def grabcut_segmentation(self, rect):
        """
        Applies the GrabCut algorithm to the loaded image.

        Args:
            rect (tuple): A rectangle (x, y, w, h) to initialize the GrabCut algorithm.

        Returns:
            np.ndarray: The segmented image using the GrabCut algorithm.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(self.image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented_image = self.image * mask2[:, :, np.newaxis]

        return segmented_image

    def detect_contours(self):
        """
        Detects contours in the loaded image.

        Returns:
            list: A list of detected contours.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edged_image = cv2.Canny(blurred_image, 50, 150)
        contours, _ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, contours):
        """
        Draws contours on the loaded image.

        Args:
            contours (list): A list of contours to draw.

        Returns:
            np.ndarray: The image with drawn contours.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        image_with_contours = self.image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        return image_with_contours

    def detect_features(self, method='SIFT'):
        """
        Detects features in the loaded image using the specified method.

        Args:
            method (str): The feature detection method to use ('SIFT', 'ORB', 'HOG').

        Returns:
            list: A list of detected keypoints.
            np.ndarray: The image with drawn keypoints.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if method == 'SIFT':
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        elif method == 'ORB':
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        elif method == 'HOG':
            hog = cv2.HOGDescriptor()
            keypoints = hog.detect(gray_image, None)
            keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in keypoints[0]]
        else:
            raise ValueError("Unknown feature detection method")

        image_with_keypoints = cv2.drawKeypoints(self.image, keypoints, None, color=(0, 255, 0))
        return keypoints, image_with_keypoints

# def main():
#     """Main function to load and display images."""
#     # Create a Tkinter root window (it will not be shown)
#     root = Tk()
#     root.withdraw()
#
#     # Open a file dialog to select one or multiple images
#     image_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
#
#     if not image_paths:
#         print("No images selected")
#         return
#
#     # Create an image loader object
#     loader = AerialImageLoader()
#
#     # Load and display each selected image
#     for image_path in image_paths:
#         loader.load_and_display(image_path)
#
#         # Print image properties
#         properties = loader.get_image_properties()
#         for key, value in properties.items():
#             print(f"{key}: {value}")
#
#         # Plot brightness histogram
#         loader.plot_brightness_histogram()
#
#         # Denoise and display the denoised image
#         denoised_image = loader.denoise_image(method='gaussian', ksize=(5, 5), sigma=1)
#         cv2.imshow('Denoised Image', denoised_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Sharpen and display the sharpened image
#         sharpened_image = loader.sharpen_image(method='unsharp_mask', amount=1.5)
#         cv2.imshow('Sharpened Image', sharpened_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Apply threshold segmentation
#         thresholded_image = loader.threshold_segmentation(threshold_value=127)
#         cv2.imshow('Threshold Segmentation', thresholded_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Apply Otsu's segmentation
#         otsu_image = loader.otsu_segmentation()
#         cv2.imshow('Otsu Segmentation', otsu_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Apply Watershed segmentation
#         watershed_image = loader.watershed_segmentation()
#         cv2.imshow('Watershed Segmentation', watershed_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Apply GrabCut segmentation
#         rect = (50, 50, 450, 290)  # Example rectangle
#         grabcut_image = loader.grabcut_segmentation(rect)
#         cv2.imshow('GrabCut Segmentation', grabcut_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Detect and draw contours
#         contours = loader.detect_contours()
#         image_with_contours = loader.draw_contours(contours)
#         cv2.imshow('Contours', image_with_contours)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Detect and display features using SIFT
#         keypoints, image_with_keypoints = loader.detect_features(method='SIFT')
#         cv2.imshow('SIFT Features', image_with_keypoints)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Detect and display features using ORB
#         keypoints, image_with_keypoints = loader.detect_features(method='ORB')
#         cv2.imshow('ORB Features', image_with_keypoints)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Detect and display features using HOG
#         keypoints, image_with_keypoints = loader.detect_features(method='HOG')
#         cv2.imshow('HOG Features', image_with_keypoints)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()