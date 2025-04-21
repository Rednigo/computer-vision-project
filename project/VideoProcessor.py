import numpy as np
import os
import cv2
import glob
from pathlib import Path

class VideoProcessor:
    """
    Class for processing video streams and detecting moving objects.
    """

    def __init__(self):
        """
        Initialize the video processor object.
        """
        self.video_capture = None
        self.current_frame = None
        self.prev_frame = None
        
        # Збалансовані параметри для виявлення рухомих об'єктів
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=750,          # Оптимальна історія для балансу
            varThreshold=24,      # Середній поріг для виявлення змін
            detectShadows=False   # Вимкнення виявлення тіней для швидкодії
        )
        
        # Зберігаємо історію кадрів для аналізу руху
        self.frame_queue = []
        self.max_queue_size = 5   # Розмір черги для аналізу

    def open_video(self, video_path):
        """
        Opens a video file or camera stream.

        Args:
            video_path: Path to the video file or camera index (0 for default camera)

        Returns:
            bool: True if the video was opened successfully, False otherwise
        """
        if isinstance(video_path, int) or os.path.exists(video_path):
            self.video_capture = cv2.VideoCapture(video_path)
            return self.video_capture.isOpened()
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    def release_video(self):
        """
        Releases the video capture object.
        """
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            
    def get_frame(self):
        """
        Gets the next frame from the video.

        Returns:
            np.ndarray: The next frame from the video or None if no more frames
        """
        if self.video_capture is None or not self.video_capture.isOpened():
            return None
            
        ret, frame = self.video_capture.read()
        if not ret:
            return None
            
        self.prev_frame = self.current_frame
        self.current_frame = frame
        
        # Оновлюємо чергу кадрів
        if len(self.frame_queue) >= self.max_queue_size:
            self.frame_queue.pop(0)  # Видаляємо найстаріший кадр
        self.frame_queue.append(frame.copy())
        
        return frame

    def get_video_properties(self):
        """
        Gets the properties of the opened video.

        Returns:
            dict: A dictionary containing video properties
        """
        if self.video_capture is None or not self.video_capture.isOpened():
            raise ValueError("No video opened")
            
        props = {
            "Width": int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "Height": int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "FPS": self.video_capture.get(cv2.CAP_PROP_FPS),
            "Frame Count": int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "Format": self.video_capture.get(cv2.CAP_PROP_FORMAT),
            "Mode": self.video_capture.get(cv2.CAP_PROP_MODE)
        }
        return props

    def apply_optical_flow(self, prev_frame, current_frame, params=None):
        """
        Applies optical flow to detect motion between two frames.

        Args:
            prev_frame: Previous frame
            current_frame: Current frame
            params: Dictionary of parameters for optical flow

        Returns:
            np.ndarray: Visualization of the optical flow
        """
        if prev_frame is None or current_frame is None:
            return None
            
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Set default parameters if not provided
        if params is None:
            params = {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, 
            current_gray, 
            None, 
            params['pyr_scale'], 
            params['levels'], 
            params['winsize'], 
            params['iterations'], 
            params['poly_n'], 
            params['poly_sigma'], 
            params['flags']
        )
        
        # Convert flow to polar coordinates (magnitude and angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV visualization
        hsv = np.zeros_like(current_frame)
        hsv[..., 1] = 255  # Saturation
        
        # Use angle for hue and magnitude for value
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
        
        # Convert HSV to BGR for visualization
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_rgb
    
    def apply_lucas_kanade_optical_flow(self, prev_frame, current_frame):
        """
        Applies Lucas-Kanade optical flow to track specific feature points.

        Args:
            prev_frame: Previous frame
            current_frame: Current frame

        Returns:
            np.ndarray: Frame with tracked points and motion vectors
        """
        if prev_frame is None or current_frame is None:
            return None
            
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Detect feature points in the previous frame
        prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        if prev_points is None:
            return current_frame.copy()
        
        # Calculate optical flow
        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, current_gray, prev_points, None, **lk_params
        )
        
        # Filter only valid points
        good_new = current_points[status == 1]
        good_old = prev_points[status == 1]
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(current_frame)
        
        # Draw the tracks
        result_frame = current_frame.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Draw line between old and new position
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            
            # Draw filled circle at the new position
            result_frame = cv2.circle(result_frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        # Add mask to the frame
        result_frame = cv2.add(result_frame, mask)
        
        return result_frame
    
    def apply_background_subtraction(self, frame, learning_rate=0.001, min_area=400, min_speed=1.5, aspect_ratio_range=(0.3, 3.5)):
        """
        Applies background subtraction to detect moving objects including aircraft.

        Args:
            frame: Current frame
            learning_rate: Learning rate for background model update
            min_area: Minimum contour area to consider
            min_speed: Minimum speed threshold (lower values detect more objects)
            aspect_ratio_range: (min, max) aspect ratio range for potential objects

        Returns:
            tuple: (Foreground mask, Frame with detected objects)
        """
        if frame is None:
            return None, None
            
        # Застосовуємо фонове віднімання з фіксованою швидкістю навчання
        fg_mask = self.background_subtractor.apply(frame, learningRate=learning_rate)
        
        # Застосовуємо морфологічні операції для усунення шуму
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((11, 11), np.uint8)
        
        # Операція відкриття (ерозія з подальшою дилатацією) для усунення дрібного шуму
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Операція закриття (дилатація з подальшою ерозією) для закриття малих отворів
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Знаходимо контури в масці переднього плану
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Малюємо прямокутники навколо виявлених об'єктів, які відповідають нашим критеріям
        result_frame = frame.copy()
        
        for contour in contours:
            # Фільтруємо за площею
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Отримуємо координати прямокутника
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w//2, y + h//2
            
            # Перевіряємо співвідношення сторін
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                continue
            
            # Використовуємо спрощену оцінку руху для балансування між виявленням літаків і ігноруванням хмар
            is_moving = True
            if len(self.frame_queue) >= 3:
                reference_frame = self.frame_queue[0]
                movement = self._simple_movement_estimation(reference_frame, frame, (x, y, w, h))
                is_moving = movement >= min_speed
            
            # Виявляємо рухомі об'єкти
            if is_moving:
                # Малюємо прямокутник
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Додаємо мітку
                label = "Moving Object"
                if aspect_ratio > 0.5 and aspect_ratio < 2.5 and area > 600:
                    label = "Aircraft"
                    
                cv2.putText(
                    result_frame, 
                    f"{label} (Area: {area:.0f})", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        
        return fg_mask, result_frame
    
    def _simple_movement_estimation(self, old_frame, current_frame, roi):
        """
        Simplified method to estimate movement between frames.
        
        Args:
            old_frame: Previous frame from history
            current_frame: Current frame
            roi: (x, y, w, h) region of interest
            
        Returns:
            float: Movement score (higher values indicate more movement)
        """
        x, y, w, h = roi
        
        # Ensure ROI is within frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(old_frame.shape[1], x + w)
        y2 = min(old_frame.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        try:
            # Extract ROIs from both frames
            old_roi = cv2.cvtColor(old_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            curr_roi = cv2.cvtColor(current_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            # Compare ROIs using absolute difference
            diff = cv2.absdiff(old_roi, curr_roi)
            
            # Calculate mean difference as movement score
            mean_diff = np.mean(diff)
            
            return mean_diff
        except Exception as e:
            return 0.0
    
    def reset_background_subtractor(self):
        """
        Resets the background subtractor model.
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=750, 
            varThreshold=24, 
            detectShadows=False
        )
        # Очищаємо чергу кадрів
        self.frame_queue = []