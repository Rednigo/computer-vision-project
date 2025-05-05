import cv2
import numpy as np
import time
import threading
import queue
from datetime import datetime
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from ObjectClassifier import ObjectClassifier
from CNNClassifier import CNNClassifier
from PreTrainedModels import PreTrainedModels

class RealTimeDetection:
    """
    Real-time aerial object detection and classification system.
    Supports multiple detection methods and live video processing.
    """
    
    def __init__(self, primary_model='yolo', backup_model='cnn'):
        """
        Initialize real-time detection system
        
        Args:
            primary_model: Primary detection model ('yolo', 'cnn', 'traditional')
            backup_model: Backup model in case primary fails
        """
        self.primary_model = primary_model
        self.backup_model = backup_model
        
        # Models
        self.pretrained_models = PreTrainedModels()
        self.cnn_classifier = None
        self.ml_classifier = None
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.detection_interval = 0.1  # seconds between detections
        self.save_detections = True
        self.save_directory = None
        
        # Real-time processing
        self.frame_queue = queue.Queue(maxsize=100)
        self.detection_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_objects': 0,
            'average_fps': 0.0,
            'detection_history': [],
            'error_count': 0
        }
        
        # Tracking
        self.object_trackers = {}
        self.tracked_objects = []
        self.next_object_id = 0
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize detection models"""
        print(f"Initializing {self.primary_model} as primary model...")
        
        if self.primary_model == 'yolo':
            try:
                self.pretrained_models.load_yolo()
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
                self._switch_to_backup()
        
        elif self.primary_model == 'cnn':
            try:
                self.cnn_classifier = CNNClassifier()
                # You'll need to load a pre-trained model here
                print("CNN model loaded successfully")
            except Exception as e:
                print(f"Failed to load CNN: {e}")
                self._switch_to_backup()
        
        elif self.primary_model == 'traditional':
            try:
                self.ml_classifier = ObjectClassifier()
                # You'll need to load a trained model here
                print("Traditional ML model loaded successfully")
            except Exception as e:
                print(f"Failed to load Traditional ML: {e}")
                self._switch_to_backup()
    
    def _switch_to_backup(self):
        """Switch to backup model if primary fails"""
        print(f"Switching to backup model: {self.backup_model}")
        old_primary = self.primary_model
        self.primary_model = self.backup_model
        self.backup_model = old_primary
        self._initialize_models()
    
    def configure(self, confidence_threshold=0.5, detection_interval=0.1, 
                 save_detections=True, save_directory=None):
        """Configure detection parameters"""
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.save_detections = save_detections
        self.save_directory = save_directory
        
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    def start_detection(self, source=0):
        """
        Start real-time detection
        
        Args:
            source: Video source (0 for webcam, video file path, or RTSP stream)
        """
        print(f"Starting real-time detection from source: {source}")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_detections)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Main capture loop
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put((time.time(), frame.copy()))
                
                # Get detection results if available
                if not self.detection_queue.empty():
                    detection_time, detections, annotated_frame = self.detection_queue.get()
                    frame = annotated_frame
                    
                    # Update statistics
                    self._update_stats(detections)
                
                # Display FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    self.stats['average_fps'] = fps
                    
                    # Draw FPS on frame
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Real-time Detection', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.stop_detection()
            cap.release()
            cv2.destroyAllWindows()
    
    def _process_detections(self):
        """Background thread for processing detections"""
        last_detection_time = 0
        
        while self.is_running:
            try:
                # Get frame from queue
                if not self.frame_queue.empty():
                    timestamp, frame = self.frame_queue.get()
                    
                    # Check if it's time for a new detection
                    if timestamp - last_detection_time >= self.detection_interval:
                        # Perform detection
                        detections = self._detect_objects(frame)
                        
                        # Annotate frame
                        annotated_frame = self._annotate_frame(frame, detections)
                        
                        # Save if enabled
                        if self.save_detections and self.save_directory:
                            self._save_detection(timestamp, frame, detections)
                        
                        # Add to detection queue
                        if not self.detection_queue.full():
                            self.detection_queue.put((timestamp, detections, annotated_frame))
                        
                        last_detection_time = timestamp
                
                else:
                    # Sleep briefly if no frames to process
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"Error in detection thread: {e}")
                self.stats['error_count'] += 1
    
    def _detect_objects(self, frame):
        """Detect objects using the current model"""
        try:
            if self.primary_model == 'yolo':
                detections, _ = self.pretrained_models.detect_yolo(
                    frame, 
                    conf_threshold=self.confidence_threshold
                )
                return detections
            
            elif self.primary_model == 'cnn':
                # For CNN, we assume the frame contains a single object to classify
                predicted_class, confidence = self.cnn_classifier.predict(frame)
                h, w = frame.shape[:2]
                return [{
                    'bbox': [0, 0, w, h],
                    'confidence': confidence,
                    'class': predicted_class
                }]
            
            elif self.primary_model == 'traditional':
                # For traditional ML, similar to CNN
                predicted_class, confidence = self.ml_classifier.predict(frame)
                h, w = frame.shape[:2]
                return [{
                    'bbox': [0, 0, w, h],
                    'confidence': confidence,
                    'class': predicted_class
                }]
            
            return []
        
        except Exception as e:
            print(f"Detection error: {e}")
            self.stats['error_count'] += 1
            return []
    
    def _annotate_frame(self, frame, detections):
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class']
            
            # Draw bounding box
            color = self._get_class_color(class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add statistics overlay
        stats_text = [
            f"Objects: {len(detections)}",
            f"Model: {self.primary_model}",
            f"Errors: {self.stats['error_count']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(annotated, text, (10, 60 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated
    
    def _get_class_color(self, class_name):
        """Get color for a class (consistent colors)"""
        # Simple hash-based color assignment
        hash_val = hash(class_name) % 360
        color = cv2.cvtColor(np.uint8([[[hash_val, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return (int(color[0]), int(color[1]), int(color[2]))
    
    def _save_detection(self, timestamp, frame, detections):
        """Save detection results to disk"""
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
        
        # Save image
        image_path = os.path.join(self.save_directory, f"detection_{timestamp_str}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Save detection data
        data_path = os.path.join(self.save_directory, f"detection_{timestamp_str}.json")
        with open(data_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'detections': detections,
                'model': self.primary_model
            }, f, indent=4)
    
    def _update_stats(self, detections):
        """Update detection statistics"""
        self.stats['total_frames'] += 1
        self.stats['detected_objects'] += len(detections)
        
        # Update detection history
        self.stats['detection_history'].append({
            'timestamp': datetime.now().isoformat(),
            'count': len(detections),
            'classes': [d['class'] for d in detections]
        })
        
        # Keep history limited
        if len(self.stats['detection_history']) > 1000:
            self.stats['detection_history'] = self.stats['detection_history'][-1000:]
    
    def stop_detection(self):
        """Stop detection system"""
        print("Stopping detection system...")
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def get_statistics(self):
        """Get current detection statistics"""
        return self.stats.copy()
    
    def generate_report(self, output_file=None):
        """Generate detection report"""
        report = {
            'summary': {
                'total_frames': self.stats['total_frames'],
                'total_detections': self.stats['detected_objects'],
                'average_fps': self.stats['average_fps'],
                'error_rate': self.stats['error_count'] / max(1, self.stats['total_frames']),
                'model_used': self.primary_model
            },
            'class_distribution': self._get_class_distribution(),
            'detection_timeline': self.stats['detection_history'][-100:]  # Last 100 detections
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report
    
    def _get_class_distribution(self):
        """Get distribution of detected classes"""
        class_counts = {}
        for entry in self.stats['detection_history']:
            for class_name in entry['classes']:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts
    
    def visualize_statistics(self, output_file=None):
        """Create visualization of detection statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Detection timeline
        if self.stats['detection_history']:
            times = [entry['timestamp'] for entry in self.stats['detection_history'][-50:]]
            counts = [entry['count'] for entry in self.stats['detection_history'][-50:]]
            
            ax1.plot(counts)
            ax1.set_title('Objects Detected Over Time')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Object Count')
            ax1.grid(True)
        
        # Class distribution
        class_dist = self._get_class_distribution()
        if class_dist:
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            ax2.bar(classes, counts)
            ax2.set_title('Class Distribution')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # Performance metrics
        metrics = {
            'FPS': self.stats['average_fps'],
            'Error Rate': self.stats['error_count'] / max(1, self.stats['total_frames']) * 100,
            'Detection Rate': self.stats['detected_objects'] / max(1, self.stats['total_frames'])
        }
        
        ax3.bar(metrics.keys(), metrics.values())
        ax3.set_title('Performance Metrics')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # Detection confidence histogram
        confidences = []
        for entry in self.stats['detection_history']:
            for det in entry.get('detections', []):
                confidences.append(det.get('confidence', 0))
        
        if confidences:
            ax4.hist(confidences, bins=20)
            ax4.set_title('Detection Confidence Distribution')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()