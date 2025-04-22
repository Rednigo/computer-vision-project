import numpy as np
import cv2
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from pathlib import Path

class ObjectClassifier:
    """
    Class for classifying aerial objects using different machine learning algorithms.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.classes = []
        self.feature_extractor = 'HOG'  # Default feature extractor
        
    def extract_features(self, image, method='HOG'):
        """
        Extract features from the image using the specified method.
        
        Args:
            image: Input image (numpy array)
            method: Feature extraction method ('HOG', 'SIFT', 'ORB')
            
        Returns:
            feature_vector: Extracted feature vector
        """
        if image is None:
            raise ValueError("No image provided for feature extraction")
            
        # Convert to grayscale if the image has 3 channels
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # Resize image to a standard size to ensure consistent feature vector size
        resized_image = cv2.resize(gray_image, (128, 128))
        
        if method == 'HOG':
            # HOG feature extraction
            hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
            feature_vector = hog.compute(resized_image)
            
        elif method == 'SIFT':
            # SIFT feature extraction
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(resized_image, None)
            
            if descriptors is None:
                # If no keypoints are detected, return a zero vector
                return np.zeros(128)
                
            # Use the mean of all descriptors as the feature vector
            feature_vector = np.mean(descriptors, axis=0) if len(descriptors) > 0 else np.zeros(128)
            
        elif method == 'ORB':
            # ORB feature extraction
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(resized_image, None)
            
            if descriptors is None:
                # If no keypoints are detected, return a zero vector
                return np.zeros(32)
                
            # Use the mean of all descriptors as the feature vector
            feature_vector = np.mean(descriptors, axis=0) if len(descriptors) > 0 else np.zeros(32)
            
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
            
        return feature_vector.flatten()
    
    def set_feature_extractor(self, method):
        """
        Set the feature extraction method.
        
        Args:
            method: Feature extraction method ('HOG', 'SIFT', 'ORB')
        """
        if method not in ['HOG', 'SIFT', 'ORB']:
            raise ValueError(f"Unsupported feature extraction method: {method}")
        
        self.feature_extractor = method
        
    def train(self, images, labels, model_type='SVM', test_size=0.2, random_state=42):
        """
        Train the classifier on the provided images and labels.
        
        Args:
            images: List of images
            labels: List of labels
            model_type: Type of model to train ('SVM', 'RandomForest', 'KNN')
            test_size: Portion of the dataset to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            accuracy: Accuracy score on the test set
            report: Classification report
        """
        if len(images) != len(labels):
            raise ValueError("Number of images and labels must match")
            
        if len(images) == 0:
            raise ValueError("No training data provided")
            
        # Extract features from all images
        features = []
        for image in images:
            feature_vector = self.extract_features(image, self.feature_extractor)
            features.append(feature_vector)
            
        features = np.array(features)
        labels = np.array(labels)
        
        # Save the unique classes
        self.classes = np.unique(labels)
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create and train the classifier
        if model_type == 'SVM':
            self.model = svm.SVC(kernel='linear', probability=True)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model_type = model_type
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict(self, image):
        """
        Predict the class of the given image.
        
        Args:
            image: Input image
            
        Returns:
            predicted_class: Predicted class
            confidence: Confidence score
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
            
        # Extract features
        feature_vector = self.extract_features(image, self.feature_extractor)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Scale the features
        feature_vector = self.scaler.transform(feature_vector)
        
        # Predict
        predicted_class = self.model.predict(feature_vector)[0]
        
        # Get confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = probabilities[list(self.model.classes_).index(predicted_class)]
        else:
            confidence = None
            
        return predicted_class, confidence
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Args:
            filename: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'classes': self.classes,
            'feature_extractor': self.feature_extractor
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Args:
            filename: Path to the saved model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
            
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
        self.feature_extractor = model_data['feature_extractor']
    
    def create_dataset_from_directory(self, dataset_dir):
        """
        Create a dataset from a directory structure.
        Assumes that the directory structure is:
        dataset_dir/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                ...
            ...
        
        Args:
            dataset_dir: Path to the dataset directory
            
        Returns:
            images: List of images
            labels: List of labels
        """
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
        
        return images, labels