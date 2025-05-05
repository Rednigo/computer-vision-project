import numpy as np
import os
import cv2
import glob
from tkinter import Tk, filedialog
from pathlib import Path
import matplotlib.pyplot as plt
from ObjectClassifier import ObjectClassifier
from CNNClassifier import CNNClassifier
from PreTrainedModels import PreTrainedModels
from DataAugmentation import DataAugmentation

class AerialImageLoader:
    """
    Class for loading and processing aerial images.
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

        self.classifier = ObjectClassifier()
        self.cnn_classifier = CNNClassifier()
        self.pretrained_models = PreTrainedModels()
        self.data_augmentation = DataAugmentation()  # Add this line

    def augment_loaded_image(self, augmentation_params=None):
        """Augment the currently loaded image"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        augmented = self.data_augmentation.augment_single_image(self.image, augmentation_params)
        return augmented

    def augment_dataset(self, input_dir, output_dir, augmentation_factor=5, 
                    augmentation_params=None):
        """Augment an entire dataset"""
        return self.data_augmentation.augment_dataset(
            input_dir, 
            output_dir, 
            augmentation_factor, 
            augmentation_params
        )

    def visualize_augmentations(self, num_samples=8, augmentation_params=None):
        """Visualize augmentation effects on current image"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        return self.data_augmentation.visualize_augmentations(
            self.image, 
            num_samples, 
            augmentation_params
        )

    def apply_specific_augmentation(self, augmentation_type, **kwargs):
        """Apply specific augmentation to current image"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        if augmentation_type == 'brightness':
            return self.data_augmentation.augment_brightness(self.image, **kwargs)
        elif augmentation_type == 'contrast':
            return self.data_augmentation.augment_contrast(self.image, **kwargs)
        elif augmentation_type == 'gaussian_noise':
            return self.data_augmentation.add_gaussian_noise(self.image, **kwargs)
        elif augmentation_type == 'salt_pepper_noise':
            return self.data_augmentation.add_salt_pepper_noise(self.image, **kwargs)
        elif augmentation_type == 'blur':
            return self.data_augmentation.apply_blur(self.image, **kwargs)
        elif augmentation_type == 'color':
            return self.data_augmentation.color_augmentation(self.image, **kwargs)
        elif augmentation_type == 'elastic':
            return self.data_augmentation.elastic_transform(self.image, **kwargs)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    def create_augmented_data_generator(self, directory, target_size=(224, 224), 
                                    batch_size=32, class_mode='categorical'):
        """Create data generator with augmentation for training"""
        return self.data_augmentation.create_keras_data_generator(
            directory, 
            target_size, 
            batch_size, 
            class_mode
        )

    def train_with_augmentation(self, dataset_dir, augmentation_factor=5, 
                            model_type='CNN', epochs=50, batch_size=32,
                            augmentation_params=None):
        """
        Train a model with augmented data
        
        Args:
            dataset_dir: Directory containing the dataset
            augmentation_factor: Number of augmented versions per image
            model_type: 'CNN' or 'Traditional'
            epochs: Number of training epochs (for CNN)
            batch_size: Batch size
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Training results
        """
        # Create temporary directory for augmented data
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Augment dataset
            print("Augmenting dataset...")
            stats = self.augment_dataset(
                dataset_dir, 
                temp_dir, 
                augmentation_factor, 
                augmentation_params
            )
            print(f"Augmentation stats: {stats}")
            
            # Train model on augmented data
            if model_type == 'CNN':
                print("Training CNN with augmented data...")
                history = self.train_cnn_classifier(
                    temp_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2
                )
                return history
            else:
                print("Training traditional model with augmented data...")
                from ObjectClassifier import ObjectClassifier
                images, labels = ObjectClassifier().create_dataset_from_directory(temp_dir)
                accuracy, report = self.classifier.train(images, labels, model_type)
                return accuracy, report

    def preview_augmentation_pipeline(self, aug_pipeline_name='medium', num_samples=9):
        """
        Preview different augmentation pipeline effects
        
        Args:
            aug_pipeline_name: 'light', 'medium', or 'heavy'
            num_samples: Number of samples to generate
            
        Returns:
            Figure showing pipeline effects
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        import matplotlib.pyplot as plt
        
        # Create pipeline
        pipeline = self.data_augmentation.advanced_augmentation_pipeline(aug_pipeline_name)
        
        # Generate samples
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Apply pipeline multiple times
        for i in range(1, num_samples):
            augmented = pipeline(image=self.image)['image']
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(augmented_rgb)
            axes[i].set_title(f"{aug_pipeline_name.capitalize()} Pipeline {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

    def compare_augmentation_effects(self, augmentation_types=None):
        """
        Compare different augmentation effects side by side
        
        Args:
            augmentation_types: List of augmentation types to compare
            
        Returns:
            Figure showing comparison
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        if augmentation_types is None:
            augmentation_types = [
                'brightness',
                'contrast',
                'gaussian_noise',
                'blur',
                'color',
                'elastic'
            ]
        
        import matplotlib.pyplot as plt
        
        n_types = len(augmentation_types)
        cols = min(3, n_types)
        rows = (n_types + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_types == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        for i, aug_type in enumerate(augmentation_types):
            augmented = self.apply_specific_augmentation(aug_type)
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(augmented_rgb)
            axes[i].set_title(aug_type.replace('_', ' ').title())
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_types, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

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

    def train_classifier(self, dataset_dir, model_type='SVM', feature_method='HOG'):
        """
        Trains a classifier on a dataset of images.
        
        Args:
            dataset_dir (str): Directory containing the dataset. The directory should have subdirectories,
                              each named after a class and containing images of that class.
            model_type (str): Type of classifier to train ('SVM', 'RandomForest', 'KNN').
            feature_method (str): Method for feature extraction ('HOG', 'SIFT', 'ORB').
            
        Returns:
            tuple: (accuracy, report) - accuracy score and classification report of the trained model.
        """
        self.classifier.set_feature_extractor(feature_method)
        images, labels = self.classifier.create_dataset_from_directory(dataset_dir)
        
        if not images:
            raise ValueError("No images found in the dataset directory")
            
        accuracy, report = self.classifier.train(images, labels, model_type)
        return accuracy, report
    
    def classify_image(self):
        """
        Classifies the loaded image using the trained classifier.
        
        Returns:
            tuple: (predicted_class, confidence) - predicted class and confidence score.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        if self.classifier.model is None:
            raise ValueError("No classifier model has been trained yet")
            
        return self.classifier.predict(self.image)
    
    def load_classifier_model(self, model_path):
        """
        Loads a trained classifier model from a file.
        
        Args:
            model_path (str): Path to the saved model file.
        """
        self.classifier.load_model(model_path)
    
    def save_classifier_model(self, model_path):
        """
        Saves the trained classifier model to a file.
        
        Args:
            model_path (str): Path to save the model file.
        """
        self.classifier.save_model(model_path)
    
    def draw_classification_result(self, predicted_class, confidence=None):
        """
        Draws the classification result on the image.
        
        Args:
            predicted_class (str): Predicted class name.
            confidence (float, optional): Confidence score of the prediction.
            
        Returns:
            np.ndarray: Image with classification result drawn on it.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        result_image = self.image.copy()
        
        # Draw a box at the top of the image
        height, width = result_image.shape[:2]
        cv2.rectangle(result_image, (0, 0), (width, 40), (0, 0, 0), -1)
        
        # Draw the class name
        text = f"Class: {predicted_class}"
        if confidence is not None:
            text += f" (Conf: {confidence:.2f})"
            
        cv2.putText(result_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image

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
        
    # Week 8: Morphological Operations
    
    def apply_erosion(self, kernel_size=5, iterations=1):
        """
        Applies erosion to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            iterations (int): Number of times erosion is applied.
            
        Returns:
            np.ndarray: The eroded image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply erosion
        eroded = cv2.erode(gray, kernel, iterations=iterations)
        
        return eroded
        
    def apply_dilation(self, kernel_size=5, iterations=1):
        """
        Applies dilation to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            iterations (int): Number of times dilation is applied.
            
        Returns:
            np.ndarray: The dilated image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply dilation
        dilated = cv2.dilate(gray, kernel, iterations=iterations)
        
        return dilated
        
    def apply_opening(self, kernel_size=5, iterations=1):
        """
        Applies opening (erosion followed by dilation) to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            iterations (int): Number of times opening is applied.
            
        Returns:
            np.ndarray: The opened image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply opening
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        return opened
        
    def apply_closing(self, kernel_size=5, iterations=1):
        """
        Applies closing (dilation followed by erosion) to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            iterations (int): Number of times closing is applied.
            
        Returns:
            np.ndarray: The closed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply closing
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return closed
        
    def apply_morphological_gradient(self, kernel_size=5):
        """
        Applies morphological gradient (dilation - erosion) to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            
        Returns:
            np.ndarray: The gradient image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply morphological gradient
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        return gradient
        
    def apply_top_hat(self, kernel_size=5):
        """
        Applies top hat transformation (original - opening) to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            
        Returns:
            np.ndarray: The top hat transformed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply top hat
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        return top_hat
        
    def apply_black_hat(self, kernel_size=5):
        """
        Applies black hat transformation (closing - original) to the loaded image.
        
        Args:
            kernel_size (int): Size of the structuring element.
            
        Returns:
            np.ndarray: The black hat transformed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply black hat
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        return black_hat
        
    def apply_custom_morphology(self, operation, kernel_shape='rect', kernel_size=5, iterations=1):
        """
        Applies a custom morphological operation with specified kernel shape.
        
        Args:
            operation (str): The morphological operation ('erode', 'dilate', 'open', 'close').
            kernel_shape (str): Shape of the kernel ('rect', 'ellipse', 'cross').
            kernel_size (int): Size of the structuring element.
            iterations (int): Number of times operation is applied.
            
        Returns:
            np.ndarray: The processed image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding if needed
        if gray.dtype != np.uint8 or np.max(gray) != 255 or np.min(gray) != 0:
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        # Create kernel based on shape
        if kernel_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            raise ValueError("Unknown kernel shape. Use 'rect', 'ellipse', or 'cross'.")
            
        # Apply selected operation
        if operation == 'erode':
            result = cv2.erode(gray, kernel, iterations=iterations)
        elif operation == 'dilate':
            result = cv2.dilate(gray, kernel, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'gradient':
            result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        elif operation == 'tophat':
            result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            raise ValueError("Unknown operation. Use 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', or 'blackhat'.")
            
        return result
        
    def enhance_segmentation_with_morphology(self, segmentation_method='threshold', morphology_operations=None, **kwargs):
        """
        Enhances segmentation results using a sequence of morphological operations.
        
        Args:
            segmentation_method (str): Method for initial segmentation ('threshold', 'otsu', 'adaptive').
            morphology_operations (list): List of dicts with operations to apply in sequence,
                                        e.g., [{'op': 'open', 'kernel_size': 5}, {'op': 'dilate', 'kernel_size': 3}]
            **kwargs: Additional parameters for segmentation.
            
        Returns:
            np.ndarray: The enhanced segmentation result.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply initial segmentation
        if segmentation_method == 'threshold':
            threshold_value = kwargs.get('threshold_value', 127)
            _, segmented = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif segmentation_method == 'otsu':
            _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif segmentation_method == 'adaptive':
            block_size = kwargs.get('block_size', 11)
            c = kwargs.get('c', 2)
            segmented = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block_size, c)
        else:
            raise ValueError("Unknown segmentation method. Use 'threshold', 'otsu', or 'adaptive'.")
            
        # If no morphological operations specified, return the segmented image
        if not morphology_operations:
            return segmented
            
        # Apply morphological operations in sequence
        result = segmented.copy()
        for operation in morphology_operations:
            op_type = operation.get('op', 'open')
            kernel_size = operation.get('kernel_size', 5)
            iterations = operation.get('iterations', 1)
            kernel_shape = operation.get('kernel_shape', 'rect')
            
            # Create kernel
            if kernel_shape == 'rect':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            elif kernel_shape == 'ellipse':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            elif kernel_shape == 'cross':
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
            # Apply operation
            if op_type == 'erode':
                result = cv2.erode(result, kernel, iterations=iterations)
            elif op_type == 'dilate':
                result = cv2.dilate(result, kernel, iterations=iterations)
            elif op_type == 'open':
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif op_type == 'close':
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif op_type == 'gradient':
                result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
            elif op_type == 'tophat':
                result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)
            elif op_type == 'blackhat':
                result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
                
        return result
        
    def remove_noise_with_morphology(self, kernel_size=3, noise_type='salt_pepper'):
        """
        Uses morphological operations to remove specific types of noise.
        
        Args:
            kernel_size (int): Size of the structuring element.
            noise_type (str): Type of noise to remove ('salt_pepper', 'speckle', 'small_holes').
            
        Returns:
            np.ndarray: The denoised image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding to ensure binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply specific noise removal
        if noise_type == 'salt_pepper':
            # For salt and pepper noise, opening removes salt noise, closing removes pepper noise
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove salt
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)  # Remove pepper
        elif noise_type == 'speckle':
            # For speckle noise, opening is effective
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif noise_type == 'small_holes':
            # For small holes, closing is effective
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError("Unknown noise type. Use 'salt_pepper', 'speckle', or 'small_holes'.")
            
        return result
        
    def extract_boundaries_with_morphology(self, kernel_size=3):
        """
        Extracts object boundaries using morphological operations.
        
        Args:
            kernel_size (int): Size of the structuring element.
            
        Returns:
            np.ndarray: Image with extracted boundaries.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Apply thresholding to ensure binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilate the binary image
        dilation = cv2.dilate(binary, kernel, iterations=1)
        
        # Subtract the binary image from its dilation to get the boundaries
        boundaries = cv2.subtract(dilation, binary)
        
        return boundaries
        
    def skeletonize_image(self):
        """
        Creates a skeleton of a binary image using morphological operations.
        
        Returns:
            np.ndarray: The skeletonized image.
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Convert to grayscale if image is color
        if len(self.image.shape) > 2:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()
            
        # Ensure binary image
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Skeletonization algorithm
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            # Step 1: Open the image
            eroded = cv2.erode(img, element)
            opened = cv2.dilate(eroded, element)
            
            # Step 2: Subtract the opened image from the original
            temp = cv2.subtract(img, opened)
            
            # Step 3: Add the temporary image to the skeleton
            skeleton = cv2.bitwise_or(skeleton, temp)
            
            # Step 4: Erode the original image
            img = eroded.copy()
            
            # Step 5: If the eroded image is empty, we're done
            if cv2.countNonZero(img) == 0:
                break
                
        return skeleton
def train_cnn_classifier(self, dataset_dir, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train CNN classifier on a dataset of images.
    
    Args:
        dataset_dir (str): Directory containing the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        
    Returns:
        history: Training history.
    """
    from pathlib import Path
    
    # Create dataset from directory
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    images = []
    labels = []
    
    # Get all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files in the class directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            image_files.extend(list(class_dir.glob(ext)))
        
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is not None:
                images.append(image)
                labels.append(class_name)
    
    # Train CNN
    history = self.cnn_classifier.train(
        images, 
        labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split
    )
    
    return history

def classify_image_cnn(self):
    """
    Classifies the loaded image using the CNN classifier.
    
    Returns:
        tuple: (predicted_class, confidence) - predicted class and confidence score.
    """
    if self.image is None:
        raise ValueError("No image loaded")
        
    if self.cnn_classifier.model is None:
        raise ValueError("CNN model has not been trained yet")
        
    return self.cnn_classifier.predict(self.image)

def load_cnn_model(self, model_path):
    """
    Loads a trained CNN classifier model from a file.
    
    Args:
        model_path (str): Path to the saved model file (without .h5 extension).
    """
    self.cnn_classifier.load_model(model_path)

def save_cnn_model(self, model_path):
    """
    Saves the trained CNN classifier model to a file.
    
    Args:
        model_path (str): Path to save the model file (without .h5 extension).
    """
    self.cnn_classifier.save_model(model_path)

def plot_cnn_training_history(self):
    """
    Plot the training history of the CNN model.
    """
    self.cnn_classifier.plot_training_history()

def get_cnn_summary(self):
    """
    Get summary of the CNN model architecture.
    
    Returns:
        summary: Model summary string.
    """
    return self.cnn_classifier.get_summary()

def load_resnet(self):
    """Load ResNet50 model"""
    self.pretrained_models.load_resnet()

def load_mobilenet(self):
    """Load MobileNetV2 model"""
    self.pretrained_models.load_mobilenet()

def load_yolo(self, model_type='yolov8n'):
    """Load YOLO model"""
    self.pretrained_models.load_yolo(model_type=model_type)

def create_unet(self, input_size=(256, 256, 3), num_classes=2):
    """Create U-Net architecture"""
    return self.pretrained_models.create_unet(input_size=input_size, num_classes=num_classes)

def load_pretrained_unet(self, model_path):
    """Load pre-trained U-Net model"""
    self.pretrained_models.load_pretrained_unet(model_path)

def save_unet(self, save_path):
    """Save U-Net model"""
    self.pretrained_models.save_unet(save_path)

def predict_with_resnet(self):
    """Predict using ResNet50"""
    if self.image_path is None:
        raise ValueError("No image loaded")
    
    return self.pretrained_models.predict_resnet(str(self.image_path))

def predict_with_mobilenet(self):
    """Predict using MobileNetV2"""
    if self.image_path is None:
        raise ValueError("No image loaded")
    
    return self.pretrained_models.predict_mobilenet(str(self.image_path))

def detect_with_yolo(self, conf_threshold=0.5):
    """Detect objects using YOLO"""
    if self.image_path is None:
        raise ValueError("No image loaded")
    
    detections, annotated_img = self.pretrained_models.detect_yolo(str(self.image_path), conf_threshold)
    
    # Update current image with annotated version
    self.image = annotated_img
    
    return detections, annotated_img

def segment_with_unet(self):
    """Segment image using U-Net"""
    if self.image_path is None:
        raise ValueError("No image loaded")
    
    mask = self.pretrained_models.segment_unet(str(self.image_path))
    
    # Create colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask > 0] = [0, 255, 0]  # Green for mask
    
    # Overlay mask on original image
    orig_img_resized = cv2.resize(self.image, (mask.shape[1], mask.shape[0]))
    overlaid = cv2.addWeighted(orig_img_resized, 0.7, colored_mask, 0.3, 0)
    
    return mask, overlaid

def compare_pretrained_models(self):
    """Compare predictions from multiple pre-trained models"""
    if self.image_path is None:
        raise ValueError("No image loaded")
    
    results = self.pretrained_models.compare_models(str(self.image_path))
    return results

def get_pretrained_model_info(self, model_name):
    """Get information about a pre-trained model"""
    return self.pretrained_models.get_model_info(model_name)

def visualize_detections(self, detections, draw_on_current=True):
    """Visualize detections on current image"""
    if draw_on_current and self.image is not None:
        annotated_img = self.pretrained_models.visualize_detections(str(self.image_path), detections)
        self.image = annotated_img
        return annotated_img
    elif self.image_path is not None:
        return self.pretrained_models.visualize_detections(str(self.image_path), detections)
    else:
        raise ValueError("No image available")