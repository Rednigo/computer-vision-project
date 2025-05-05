import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
import rasterio
from rasterio.windows import Window
import shapely.geometry as geom
from shapely.ops import unary_union
import geopandas as gpd
import fiona
import json
import os
from datetime import datetime
from PreTrainedModels import PreTrainedModels
from CNNClassifier import CNNClassifier

class SatelliteImageAnalysis:
    """
    Satellite image analysis system for detecting and classifying aerial objects.
    Processes large satellite/aerial images with tiling for efficient detection.
    """
    
    def __init__(self, tile_size=512, overlap=0.2):
        """
        Initialize satellite image analysis system
        
        Args:
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles (0-1)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        
        # Detection models
        self.yolo_detector = None
        self.cnn_classifier = None
        self.current_model = 'yolo'
        
        # Results storage
        self.detections = []
        self.detection_map = None
        self.heat_map = None
        
        # Geospatial data
        self.geotransform = None
        self.projection = None
        self.bounds = None
        
        # Initialize models
        self._setup_models()
    
    def _setup_models(self):
        """Setup detection models"""
        print("Initializing detection models...")
        self.pretrained_models = PreTrainedModels()
        
        # Load YOLO for object detection
        try:
            self.pretrained_models.load_yolo()
            self.yolo_detector = self.pretrained_models
            print("YOLO detector loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
        
        # Load CNN for classification
        try:
            self.cnn_classifier = CNNClassifier()
            print("CNN classifier loaded successfully")
        except Exception as e:
            print(f"Failed to load CNN: {e}")
    
    def load_satellite_image(self, image_path):
        """
        Load satellite/aerial image and extract metadata
        
        Args:
            image_path: Path to image file
            
        Returns:
            image: Loaded image array
        """
        if str(image_path).endswith(('.tif', '.tiff')):
            # Handle GeoTIFF files
            with rasterio.open(image_path) as src:
                image = src.read([1, 2, 3]).transpose(1, 2, 0)  # RGB bands
                self.geotransform = src.transform
                self.projection = src.crs
                self.bounds = src.bounds
                
                # Convert to 8-bit if needed
                if image.dtype != np.uint8:
                    image = self._convert_to_8bit(image)
                
                return image
        else:
            # Handle regular image files
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract EXIF geolocation if available
            self._extract_geolocation(image_path)
            
            return image
    
    def _convert_to_8bit(self, image):
        """Convert image to 8-bit representation"""
        # Normalize each band separately
        normalized = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):
            band = image[:, :, i]
            band_min, band_max = np.percentile(band, (2, 98))
            normalized[:, :, i] = np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255)
        return normalized
    
    def _extract_geolocation(self, image_path):
        """Extract geolocation from image EXIF data"""
        try:
            # This is a simplified version - you might want to use more robust EXIF reading
            import exifread
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                # Extract GPS coordinates if available
                # Implementation depends on specific EXIF format
        except Exception as e:
            print(f"Could not extract geolocation: {e}")
    
    def create_tiles(self, image):
        """
        Create overlapping tiles from large image
        
        Args:
            image: Large image to tile
            
        Returns:
            tiles: List of (tile, x_offset, y_offset)
        """
        tiles = []
        height, width = image.shape[:2]
        
        for y in range(0, height - self.tile_size + 1, self.stride):
            for x in range(0, width - self.tile_size + 1, self.stride):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append((tile, x, y))
        
        # Handle edge tiles
        if width % self.stride != 0:
            for y in range(0, height - self.tile_size + 1, self.stride):
                tile = image[y:y+self.tile_size, -self.tile_size:]
                tiles.append((tile, width - self.tile_size, y))
        
        if height % self.stride != 0:
            for x in range(0, width - self.tile_size + 1, self.stride):
                tile = image[-self.tile_size:, x:x+self.tile_size]
                tiles.append((tile, x, height - self.tile_size))
        
        # Corner tile
        if width % self.stride != 0 and height % self.stride != 0:
            tile = image[-self.tile_size:, -self.tile_size:]
            tiles.append((tile, width - self.tile_size, height - self.tile_size))
        
        return tiles
    
    def detect_objects_in_tiles(self, tiles, confidence_threshold=0.5):
        """
        Detect objects in image tiles
        
        Args:
            tiles: List of tiles to process
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            detections: List of detections with global coordinates
        """
        all_detections = []
        
        for idx, (tile, x_offset, y_offset) in enumerate(tiles):
            if self.current_model == 'yolo':
                tile_detections = self._detect_with_yolo(tile, confidence_threshold)
            else:
                tile_detections = self._detect_with_cnn(tile, confidence_threshold)
            
            # Convert tile coordinates to global coordinates
            for det in tile_detections:
                det['global_bbox'] = [
                    det['bbox'][0] + x_offset,
                    det['bbox'][1] + y_offset,
                    det['bbox'][2] + x_offset,
                    det['bbox'][3] + y_offset
                ]
                det['tile_id'] = idx
                all_detections.append(det)
        
        return all_detections
    
    def _detect_with_yolo(self, tile, confidence_threshold):
        """Detect objects using YOLO"""
        if self.yolo_detector is None:
            return []
        
        # Convert RGB to BGR for YOLO
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        
        detections, _ = self.yolo_detector.detect_yolo(
            tile_bgr,
            conf_threshold=confidence_threshold
        )
        
        return detections
    
    def _detect_with_cnn(self, tile, confidence_threshold):
        """Detect/classify using CNN"""
        if self.cnn_classifier is None:
            return []
        
        predicted_class, confidence = self.cnn_classifier.predict(tile)
        
        if confidence >= confidence_threshold:
            return [{
                'bbox': [0, 0, tile.shape[1], tile.shape[0]],
                'confidence': confidence,
                'class': predicted_class
            }]
        
        return []
    
    def merge_overlapping_detections(self, detections, iou_threshold=0.5):
        """
        Merge overlapping detections using Non-Maximum Suppression
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for merging
            
        Returns:
            merged_detections: List of merged detections
        """
        if not detections:
            return []
        
        # Group detections by class
        grouped = {}
        for det in detections:
            cls = det['class']
            if cls not in grouped:
                grouped[cls] = []
            grouped[cls].append(det)
        
        merged_detections = []
        
        for cls, class_detections in grouped.items():
            # Convert to format for NMS
            boxes = np.array([det['global_bbox'] for det in class_detections])
            scores = np.array([det['confidence'] for det in class_detections])
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            
            # Keep selected detections
            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    merged_detections.append(class_detections[idx])
        
        return merged_detections
    
    def create_detection_map(self, image, detections):
        """
        Create visual detection map
        
        Args:
            image: Original image
            detections: List of detections
            
        Returns:
            detection_map: Annotated image
        """
        detection_map = image.copy()
        
        # Color mapping for different classes
        class_colors = {}
        
        for det in detections:
            x1, y1, x2, y2 = det['global_bbox']
            confidence = det['confidence']
            class_name = det['class']
            
            # Get or assign color for class
            if class_name not in class_colors:
                # Generate unique color for each class
                hue = hash(class_name) % 360
                color_hsv = np.uint8([[[hue, 255, 255]]])
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
                class_colors[class_name] = tuple(map(int, color_rgb))
            
            color = class_colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(detection_map, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(detection_map, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            cv2.putText(detection_map, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return detection_map
    
    def create_heat_map(self, image_shape, detections, cell_size=50):
        """
        Create detection density heat map
        
        Args:
            image_shape: Shape of the original image
            detections: List of detections
            cell_size: Size of heat map cells
            
        Returns:
            heat_map: Heat map visualization
        """
        height, width = image_shape[:2]
        grid_height = height // cell_size
        grid_width = width // cell_size
        
        heat_grid = np.zeros((grid_height, grid_width))
        
        # Count detections in each grid cell
        for det in detections:
            x1, y1, x2, y2 = det['global_bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            grid_x = min(center_x // cell_size, grid_width - 1)
            grid_y = min(center_y // cell_size, grid_height - 1)
            
            heat_grid[grid_y, grid_x] += 1
        
        # Create visualization
        heat_map = cv2.resize(heat_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        
        return heat_map
    
    def export_geospatial_results(self, detections, output_path):
        """
        Export detection results as geospatial data
        
        Args:
            detections: List of detections
            output_path: Path to save results
        """
        if self.geotransform is None:
            print("No geospatial information available")
            return
        
        # Convert pixel coordinates to geographic coordinates
        features = []
        
        for det in detections:
            x1, y1, x2, y2 = det['global_bbox']
            
            # Convert to geographic coordinates