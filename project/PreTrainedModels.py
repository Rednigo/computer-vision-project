import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_decode
import numpy as np
import cv2
import ultralytics
from ultralytics import YOLO
import torch
import os
from pathlib import Path

class PreTrainedModels:
    """
    Class for working with pre-trained models:
    - ResNet50
    - MobileNetV2
    - YOLOv8 (for object detection)
    - U-Net (for segmentation)
    """
    
    def __init__(self):
        """Initialize pre-trained models"""
        self.resnet_model = None
        self.mobilenet_model = None
        self.yolo_model = None
        self.unet_model = None
        self.current_model = None
    
    def load_resnet(self):
        """Load ResNet50 model"""
        print("Loading ResNet50...")
        self.resnet_model = ResNet50(weights='imagenet')
        self.current_model = 'resnet'
        print("ResNet50 loaded successfully")
    
    def load_mobilenet(self):
        """Load MobileNetV2 model"""
        print("Loading MobileNetV2...")
        self.mobilenet_model = MobileNetV2(weights='imagenet')
        self.current_model = 'mobilenet'
        print("MobileNetV2 loaded successfully")
    
    def load_yolo(self, model_type='yolov8n'):
        """
        Load YOLO model
        Args:
            model_type: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', or 'yolov8x'
        """
        print(f"Loading {model_type}...")
        try:
            self.yolo_model = YOLO(f'{model_type}.pt')
            self.current_model = 'yolo'
            print(f"{model_type} loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            # Try to download the model
            from ultralytics import download
            download(f"{model_type}.pt")
            self.yolo_model = YOLO(f'{model_type}.pt')
            self.current_model = 'yolo'
    
    def create_unet(self, input_size=(256, 256, 3), num_classes=2):
        """
        Create U-Net architecture for segmentation
        Args:
            input_size: Input image dimensions
            num_classes: Number of segmentation classes
        """
        inputs = tf.keras.Input(input_size)
        
        # Encoder
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Bridge
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        
        # Decoder
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output layer
        if num_classes == 2:
            outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        else:
            outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.unet_model = model
        self.current_model = 'unet'
        print("U-Net created successfully")
        return model
    
    def predict_resnet(self, image_path):
        """
        Predict using ResNet50
        Args:
            image_path: Path to image
        Returns:
            List of predictions with (class, description, confidence)
        """
        if self.resnet_model is None:
            self.load_resnet()
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet_preprocess(x)
        
        # Make prediction
        preds = self.resnet_model.predict(x)
        decoded_preds = resnet_decode(preds, top=5)[0]
        
        # Format results
        results = []
        for i, (imagenet_id, description, score) in enumerate(decoded_preds):
            results.append((imagenet_id, description, float(score)))
        
        return results
    
    def predict_mobilenet(self, image_path):
        """
        Predict using MobileNetV2
        Args:
            image_path: Path to image
        Returns:
            List of predictions with (class, description, confidence)
        """
        if self.mobilenet_model is None:
            self.load_mobilenet()
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = mobilenet_preprocess(x)
        
        # Make prediction
        preds = self.mobilenet_model.predict(x)
        decoded_preds = mobilenet_decode(preds, top=5)[0]
        
        # Format results
        results = []
        for i, (imagenet_id, description, score) in enumerate(decoded_preds):
            results.append((imagenet_id, description, float(score)))
        
        return results
    
    def detect_yolo(self, image_path, conf_threshold=0.5):
        """
        Detect objects using YOLO
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detections
        Returns:
            Detection results and annotated image
        """
        if self.yolo_model is None:
            self.load_yolo()
        
        # Load image
        img = cv2.imread(image_path)
        
        # Make prediction
        results = self.yolo_model(img, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # Get class name
                class_name = self.yolo_model.names[int(cls)]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name
                })
        
        # Annotate image
        annotated_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return detections, annotated_img
    
    def segment_unet(self, image_path):
        """
        Segment image using U-Net
        Args:
            image_path: Path to image
        Returns:
            Segmentation mask
        """
        if self.unet_model is None:
            self.create_unet()
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        mask = self.unet_model.predict(img)
        mask = mask[0, :, :, 0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        return mask
    
    def load_pretrained_unet(self, model_path):
        """
        Load a pre-trained U-Net model
        Args:
            model_path: Path to saved model
        """
        self.unet_model = tf.keras.models.load_model(model_path)
        self.current_model = 'unet'
        print(f"U-Net loaded from {model_path}")
    
    def save_unet(self, save_path):
        """
        Save U-Net model
        Args:
            save_path: Path to save model
        """
        if self.unet_model is None:
            raise ValueError("No U-Net model to save")
        
        self.unet_model.save(save_path)
        print(f"U-Net saved to {save_path}")
    
    def get_model_info(self, model_name):
        """
        Get information about the model
        Args:
            model_name: 'resnet', 'mobilenet', 'yolo', or 'unet'
        Returns:
            Model information
        """
        if model_name == 'resnet':
            if self.resnet_model is None:
                return "ResNet50 not loaded"
            summary = []
            self.resnet_model.summary(print_fn=lambda x: summary.append(x))
            return '\n'.join(summary)
        
        elif model_name == 'mobilenet':
            if self.mobilenet_model is None:
                return "MobileNetV2 not loaded"
            summary = []
            self.mobilenet_model.summary(print_fn=lambda x: summary.append(x))
            return '\n'.join(summary)
        
        elif model_name == 'yolo':
            if self.yolo_model is None:
                return "YOLO not loaded"
            return f"YOLO Model Info:\n{self.yolo_model.info()}"
        
        elif model_name == 'unet':
            if self.unet_model is None:
                return "U-Net not loaded"
            summary = []
            self.unet_model.summary(print_fn=lambda x: summary.append(x))
            return '\n'.join(summary)
        
        else:
            return f"Unknown model: {model_name}"
    
    def visualize_detections(self, image_path, detections):
        """
        Visualize object detections
        Args:
            image_path: Path to original image
            detections: Detection results
        Returns:
            Annotated image
        """
        img = cv2.imread(image_path)
        
        for det in detections:
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if 'class' in det and 'confidence' in det:
                    label = f"{det['class']}: {det['confidence']:.2f}"
                    cv2.putText(img, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img
    
    def compare_models(self, image_path):
        """
        Compare predictions from multiple models
        Args:
            image_path: Path to image
        Returns:
            Comparison results
        """
        results = {}
        
        # ResNet prediction
        try:
            results['resnet'] = self.predict_resnet(image_path)
        except Exception as e:
            results['resnet'] = f"Error: {str(e)}"
        
        # MobileNet prediction
        try:
            results['mobilenet'] = self.predict_mobilenet(image_path)
        except Exception as e:
            results['mobilenet'] = f"Error: {str(e)}"
        
        # YOLO detection
        try:
            detections, _ = self.detect_yolo(image_path)
            results['yolo'] = detections
        except Exception as e:
            results['yolo'] = f"Error: {str(e)}"
        
        return results