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

    # Week 7: Geometric Transformations

    def resize_image(self, width=None, height=None, scale=None, method=cv2.INTER_LINEAR):
        """
        Resizes the loaded image.

        Args:
            width (int, optional): The target width in pixels.
            height (int, optional): The target height in pixels.
            scale (float, optional): Scale factor for both dimensions.
            method: Interpolation method to use (default: cv2.INTER_LINEAR).

        Returns:
            np.ndarray: The resized image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if scale is not None:
            new_width = int(self.image.shape[1] * scale)
            new_height = int(self.image.shape[0] * scale)
        elif width is not None and height is not None:
            new_width = width
            new_height = height
        elif width is not None:
            aspect_ratio = width / float(self.image.shape[1])
            new_width = width
            new_height = int(self.image.shape[0] * aspect_ratio)
        elif height is not None:
            aspect_ratio = height / float(self.image.shape[0])
            new_height = height
            new_width = int(self.image.shape[1] * aspect_ratio)
        else:
            raise ValueError("Either scale, width, height, or both width and height must be specified")

        resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=method)
        return resized_image

    def rotate_image(self, angle, scale=1.0, center=None):
        """
        Rotates the loaded image.

        Args:
            angle (float): Rotation angle in degrees.
            scale (float, optional): Scaling factor.
            center (tuple, optional): Center of rotation. If None, the center of the image is used.

        Returns:
            np.ndarray: The rotated image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        h, w = self.image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Calculate the new bounds to ensure the entire rotated image is visible
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the actual rotation
        rotated_image = cv2.warpAffine(self.image, M, (new_w, new_h))
        return rotated_image

    def affine_transform(self, src_points, dst_points):
        """
        Applies an affine transformation to the loaded image.

        Args:
            src_points (np.array): Source points in the input image (3 points).
            dst_points (np.array): Destination points in the output image (3 points).

        Returns:
            np.ndarray: The transformed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if len(src_points) != 3 or len(dst_points) != 3:
            raise ValueError("Affine transformation requires exactly 3 points")

        height, width = self.image.shape[:2]

        # Calculate the transformation matrix
        M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))

        # Apply the affine transformation
        affine_img = cv2.warpAffine(self.image, M, (width, height))
        return affine_img

    def perspective_transform(self, src_points, dst_points):
        """
        Applies a perspective transformation to the loaded image.

        Args:
            src_points (np.array): Source points in the input image (4 points).
            dst_points (np.array): Destination points in the output image (4 points).

        Returns:
            np.ndarray: The transformed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("Perspective transformation requires exactly 4 points")

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))

        # Calculate the output size
        min_x = min(dst_points[:, 0])
        max_x = max(dst_points[:, 0])
        min_y = min(dst_points[:, 1])
        max_y = max(dst_points[:, 1])
        
        output_size = (int(max_x - min_x), int(max_y - min_y))
        
        # Apply the perspective transformation
        warped_img = cv2.warpPerspective(self.image, M, output_size)
        return warped_img

    def crop_image(self, x, y, width, height):
        """
        Crops a region from the loaded image.

        Args:
            x (int): X-coordinate of the top-left corner.
            y (int): Y-coordinate of the top-left corner.
            width (int): Width of the region.
            height (int): Height of the region.

        Returns:
            np.ndarray: The cropped image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Ensure crop region is within image bounds
        h, w = self.image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        width = min(w - x, width)
        height = min(h - y, height)

        cropped_img = self.image[y:y+height, x:x+width]
        return cropped_img

    def flip_image(self, flip_code):
        """
        Flips the loaded image.

        Args:
            flip_code (int): 0 for flipping around the x-axis (vertically),
                            1 for flipping around the y-axis (horizontally),
                            -1 for flipping around both axes.

        Returns:
            np.ndarray: The flipped image.
        """
        if self.image is None:
            raise ValueError("No image loaded")

        flipped_img = cv2.flip(self.image, flip_code)
        return flipped_img