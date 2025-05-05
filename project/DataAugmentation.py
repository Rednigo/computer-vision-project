import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random
import os
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class DataAugmentation:
    """
    Class for data augmentation operations on aerial image datasets.
    Supports various augmentation techniques for improving model accuracy.
    """
    
    def __init__(self):
        """Initialize data augmentation pipelines"""
        self.albumentations_pipeline = None
        self.keras_generator = None
        self.setup_pipelines()
    
    def setup_pipelines(self):
        """Setup augmentation pipelines"""
        # Albumentations pipeline for geometric transforms
        self.albumentations_pipeline = A.Compose([
            A.Rotate(limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.8),
        ])
        
        # Keras ImageDataGenerator for real-time augmentation
        self.keras_generator = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    
    def augment_brightness(self, image, brightness_range=(0.7, 1.3)):
        """
        Augment image brightness
        
        Args:
            image: Input image
            brightness_range: Range of brightness factors
            
        Returns:
            Augmented image
        """
        factor = random.uniform(*brightness_range)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def augment_contrast(self, image, contrast_range=(0.7, 1.3)):
        """
        Augment image contrast
        
        Args:
            image: Input image
            contrast_range: Range of contrast factors
            
        Returns:
            Augmented image
        """
        if len(image.shape) == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image)
        
        factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(image_pil)
        enhanced = enhancer.enhance(factor)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        else:
            return np.array(enhanced)
    
    def add_gaussian_noise(self, image, mean=0, std=25):
        """
        Add Gaussian noise to image
        
        Args:
            image: Input image
            mean: Mean of noise distribution
            std: Standard deviation of noise
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """
        Add salt and pepper noise to image
        
        Args:
            image: Input image
            salt_prob: Probability of salt noise
            pepper_prob: Probability of pepper noise
            
        Returns:
            Noisy image
        """
        output = image.copy()
        
        # Salt noise
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        output[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        output[pepper_mask] = 0
        
        return output
    
    def apply_blur(self, image, blur_type='gaussian', kernel_size=5):
        """
        Apply different types of blur
        
        Args:
            image: Input image
            blur_type: 'gaussian', 'motion', or 'average'
            kernel_size: Size of blur kernel
            
        Returns:
            Blurred image
        """
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            return cv2.filter2D(image, -1, kernel)
        elif blur_type == 'average':
            return cv2.blur(image, (kernel_size, kernel_size))
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")
    
    def color_augmentation(self, image, hue_shift=20, sat_shift=30, val_shift=30):
        """
        Augment image colors in HSV space
        
        Args:
            image: Input image
            hue_shift: Maximum hue shift
            sat_shift: Maximum saturation shift
            val_shift: Maximum value shift
            
        Returns:
            Color augmented image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue shift
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-hue_shift, hue_shift)) % 180
        
        # Saturation shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.uniform(-sat_shift, sat_shift), 0, 255)
        
        # Value shift
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.uniform(-val_shift, val_shift), 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def elastic_transform(self, image, alpha=100, sigma=10):
        """
        Apply elastic deformation to image
        
        Args:
            image: Input image
            alpha: Scaling factor for deformation
            sigma: Smoothing parameter
            
        Returns:
            Elastically transformed image
        """
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x_displaced = x + dx
        y_displaced = y + dy
        
        # Apply transformation
        return cv2.remap(image, x_displaced.astype(np.float32), y_displaced.astype(np.float32), cv2.INTER_LINEAR)
    
    def augment_single_image(self, image, augmentation_params=None):
        """
        Apply comprehensive augmentation to a single image
        
        Args:
            image: Input image
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Augmented image
        """
        if augmentation_params is None:
            augmentation_params = {
                'geometry': True,
                'brightness': True,
                'contrast': True,
                'noise': True,
                'blur': True,
                'color': True,
                'elastic': False  # Disabled by default as it's intensive
            }
        
        augmented = image.copy()
        
        # Geometric transformations
        if augmentation_params.get('geometry', False):
            augmented = self.albumentations_pipeline(image=augmented)['image']
        
        # Brightness augmentation
        if augmentation_params.get('brightness', False) and random.random() > 0.5:
            augmented = self.augment_brightness(augmented)
        
        # Contrast augmentation
        if augmentation_params.get('contrast', False) and random.random() > 0.5:
            augmented = self.augment_contrast(augmented)
        
        # Noise augmentation
        if augmentation_params.get('noise', False):
            if random.random() > 0.7:  # 30% chance of gaussian noise
                augmented = self.add_gaussian_noise(augmented)
            elif random.random() > 0.8:  # 20% chance of salt-pepper noise
                augmented = self.add_salt_pepper_noise(augmented)
        
        # Blur augmentation
        if augmentation_params.get('blur', False) and random.random() > 0.7:
            blur_type = random.choice(['gaussian', 'motion', 'average'])
            augmented = self.apply_blur(augmented, blur_type=blur_type)
        
        # Color augmentation
        if augmentation_params.get('color', False) and random.random() > 0.5:
            augmented = self.color_augmentation(augmented)
        
        # Elastic transformation
        if augmentation_params.get('elastic', False) and random.random() > 0.8:
            augmented = self.elastic_transform(augmented)
        
        return augmented
    
    def augment_dataset(self, input_dir, output_dir, augmentation_factor=5, 
                       augmentation_params=None):
        """
        Augment an entire dataset
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
            augmentation_factor: Number of augmented versions per original image
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Statistics of augmentation process
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'original_images': 0,
            'augmented_images': 0,
            'failed_images': 0
        }
        
        # Process each class directory
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_output_dir = output_path / class_name
                class_output_dir.mkdir(exist_ok=True)
                
                print(f"Processing class: {class_name}")
                
                # Process images in each class
                image_files = [f for f in class_dir.glob('*') 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                for img_file in image_files:
                    try:
                        # Load image
                        image = cv2.imread(str(img_file))
                        if image is None:
                            stats['failed_images'] += 1
                            continue
                        
                        # Save original image
                        original_output = class_output_dir / f"orig_{img_file.name}"
                        cv2.imwrite(str(original_output), image)
                        stats['original_images'] += 1
                        
                        # Generate augmented versions
                        for i in range(augmentation_factor):
                            augmented = self.augment_single_image(image, augmentation_params)
                            
                            # Save augmented image
                            aug_filename = f"aug_{i}_{img_file.name}"
                            aug_output = class_output_dir / aug_filename
                            cv2.imwrite(str(aug_output), augmented)
                            stats['augmented_images'] += 1
                        
                        print(f"Processed: {img_file.name}")
                        
                    except Exception as e:
                        print(f"Error processing {img_file}: {str(e)}")
                        stats['failed_images'] += 1
        
        return stats
    
    def visualize_augmentations(self, image, num_samples=8, augmentation_params=None):
        """
        Visualize different augmentation effects on an image
        
        Args:
            image: Input image
            num_samples: Number of augmented samples to generate
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Figure showing original and augmented images
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Generate augmented samples
        for i in range(1, num_samples + 1):
            augmented = self.augment_single_image(image, augmentation_params)
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(augmented_rgb)
            axes[i].set_title(f"Augmented {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_keras_data_generator(self, directory, target_size=(224, 224), 
                                   batch_size=32, class_mode='categorical'):
        """
        Create Keras data generator with augmentation
        
        Args:
            directory: Dataset directory
            target_size: Target image size
            batch_size: Batch size
            class_mode: Classification mode
            
        Returns:
            Data generator
        """
        return self.keras_generator.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True
        )
    
    def augment_for_object_detection(self, image, bboxes, category_ids):
        """
        Apply augmentation for object detection (preserves bounding boxes)
        
        Args:
            image: Input image
            bboxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            category_ids: List of category IDs for each bbox
            
        Returns:
            Augmented image and updated bboxes
        """
        # Albumentations transform for object detection
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        
        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    
    def advanced_augmentation_pipeline(self, severity='medium'):
        """
        Create an advanced augmentation pipeline
        
        Args:
            severity: 'light', 'medium', or 'heavy'
            
        Returns:
            Albumentations transform
        """
        if severity == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.5),
            ])
        elif severity == 'medium':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ])
        elif severity == 'heavy':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.Rotate(limit=45, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.3),
                A.ElasticTransform(alpha=50, sigma=5, p=0.2),
                A.RandomFog(p=0.1),
                A.RandomShadow(p=0.2),
            ])
        else:
            raise ValueError(f"Unknown severity level: {severity}")