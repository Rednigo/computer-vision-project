import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import cv2
import os
import pickle
from pathlib import Path

class CNNClassifier:
    """
    Convolutional Neural Network classifier for aerial object recognition.
    """
    
    def __init__(self, image_size=(128, 128)):
        """
        Initialize CNN classifier.
        
        Args:
            image_size: Target size for input images (height, width)
        """
        self.image_size = image_size
        self.model = None
        self.classes = []
        self.history = None
    
    def create_model(self, num_classes):
        """
        Create CNN model architecture.
        
        Args:
            num_classes: Number of classes for classification
            
        Returns:
            model: Created Keras model
        """
        model = Sequential([
            # First convolution block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second convolution block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third convolution block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess single image for CNN input.
        
        Args:
            image: Input image
            
        Returns:
            preprocessed_image: Processed image ready for CNN
        """
        # Resize image
        img = cv2.resize(image, self.image_size)
        
        # Convert to RGB if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    
    def prepare_dataset(self, images, labels):
        """
        Prepare dataset for CNN training.
        
        Args:
            images: List of images
            labels: List of labels
            
        Returns:
            X: Preprocessed images
            y: One-hot encoded labels
            classes: Unique class names
        """
        X = []
        self.classes = np.unique(labels)
        
        # One-hot encode labels
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        y = []
        
        for image, label in zip(images, labels):
            # Preprocess image
            processed_img = self.preprocess_image(image)
            X.append(processed_img)
            
            # One-hot encode label
            one_hot = np.zeros(len(self.classes))
            one_hot[class_to_idx[label]] = 1
            y.append(one_hot)
        
        return np.array(X), np.array(y)
    
    def train(self, images, labels, epochs=50, batch_size=32, validation_split=0.2, augmentation=True):
        """
        Train CNN classifier.
        
        Args:
            images: List of training images
            labels: List of training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            augmentation: Whether to use data augmentation
            
        Returns:
            history: Training history
        """
        # Prepare dataset
        X, y = self.prepare_dataset(images, labels)
        
        # Create model if not already created
        if self.model is None:
            self.create_model(len(self.classes))
        
        # Data augmentation
        if augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                shear_range=0.1,
                fill_mode='nearest'
            )
            
            # Fit the generator on training data
            datagen.fit(X)
            
            # Create data generators
            train_generator = datagen.flow(X, y, batch_size=batch_size)
            validation_split_idx = int(len(X) * (1 - validation_split))
            
            train_x, val_x = X[:validation_split_idx], X[validation_split_idx:]
            train_y, val_y = y[:validation_split_idx], y[validation_split_idx:]
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            checkpoint = ModelCheckpoint('cnn_model_best.h5', monitor='val_accuracy', save_best_only=True)
            
            # Train with augmentation
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=(val_x, val_y),
                callbacks=[early_stopping, checkpoint]
            )
        else:
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            checkpoint = ModelCheckpoint('cnn_model_best.h5', monitor='val_accuracy', save_best_only=True)
            
            # Train without augmentation
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, checkpoint]
            )
        
        self.history = history
        return history
    
    def predict(self, image):
        """
        Predict class of input image.
        
        Args:
            image: Input image
            
        Returns:
            predicted_class: Predicted class name
            confidence: Confidence score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess image
        processed_img = self.preprocess_image(image)
        img_array = np.expand_dims(processed_img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        predicted_class = self.classes[class_idx]
        
        return predicted_class, confidence
    
    def save_model(self, filepath):
        """
        Save the entire CNN model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model architecture and weights
        self.model.save(filepath + '.h5')
        
        # Save additional metadata
        metadata = {
            'classes': self.classes,
            'image_size': self.image_size,
            'history': self.history.history if self.history else None
        }
        
        with open(filepath + '_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_model(self, filepath):
        """
        Load a saved CNN model.
        
        Args:
            filepath: Path to the saved model
        """
        # Load model
        self.model = tf.keras.models.load_model(filepath + '.h5')
        
        # Load metadata
        with open(filepath + '_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.classes = metadata['classes']
            self.image_size = metadata['image_size']
            self.history = metadata['history']
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        import matplotlib.pyplot as plt
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """
        Get model summary.
        
        Returns:
            summary: Model summary string
        """
        if self.model is None:
            return "No model created yet"
        
        summary_string = []
        self.model.summary(print_fn=lambda x: summary_string.append(x))
        return '\n'.join(summary_string)