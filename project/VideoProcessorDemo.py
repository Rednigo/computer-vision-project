import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
from VideoProcessor import VideoProcessor

def main():
    """Demonstration of VideoProcessor functionality with balanced sensitivity."""
    # Create Tkinter root (will not be shown)
    root = tk.Tk()
    root.withdraw()
    
    # Create video processor
    video_processor = VideoProcessor()
    
    # Ask user to select a video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
    )
    
    if not video_path:
        print("No video selected. Exiting...")
        return
    
    # Open video
    if not video_processor.open_video(video_path):
        print(f"Failed to open video: {video_path}")
        return
    
    # Get and print video properties
    props = video_processor.get_video_properties()
    print("\nVideo Properties:")
    for key, value in props.items():
        print(f"{key}: {value}")
    
    # Create windows for different views
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 640, 480)
    
    cv2.namedWindow("Optical Flow (Farneback)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow (Farneback)", 640, 480)
    
    cv2.namedWindow("Optical Flow (Lucas-Kanade)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optical Flow (Lucas-Kanade)", 640, 480)
    
    cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Background Subtraction", 640, 480)
    
    # Process video frames
    prev_frame = None
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Get next frame
        frame = video_processor.get_frame()
        if frame is None:
            break
        
        # Update frame count for FPS calculation
        frame_count += 1
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = "FPS: N/A"
        
        # Display original frame with FPS
        original_with_fps = frame.copy()
        cv2.putText(
            original_with_fps, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), 
            2
        )
        cv2.imshow("Original", original_with_fps)
        
        # Apply and display optical flow only if we have previous frame
        if prev_frame is not None:
            # Farneback dense optical flow
            flow_frame = video_processor.apply_optical_flow(prev_frame, frame)
            if flow_frame is not None:
                cv2.putText(
                    flow_frame, 
                    fps_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 255), 
                    2
                )
                cv2.imshow("Optical Flow (Farneback)", flow_frame)
            
            # Lucas-Kanade sparse optical flow
            lk_frame = video_processor.apply_lucas_kanade_optical_flow(prev_frame, frame)
            if lk_frame is not None:
                cv2.putText(
                    lk_frame, 
                    fps_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 255), 
                    2
                )
                cv2.imshow("Optical Flow (Lucas-Kanade)", lk_frame)
        
        # Apply optimized background subtraction with balanced settings
        _, bg_sub_frame = video_processor.apply_background_subtraction(
            frame, 
            learning_rate=0.001,  # Balanced learning rate
            min_area=400,        # Lower minimum area to detect more objects
            min_speed=1.5,       # Lower speed threshold for better detection
            aspect_ratio_range=(0.3, 3.5)  # Wider aspect ratio range
        )
        
        if bg_sub_frame is not None:
            # Add FPS text to the frame
            cv2.putText(
                bg_sub_frame, 
                fps_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )
            cv2.imshow("Background Subtraction", bg_sub_frame)
        
        # Store current frame for next iteration
        prev_frame = frame.copy()
        
        # Wait for key press (exit on 'q' or ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC
            break
        
        # Reset background subtractor if 'r' is pressed
        if key == ord('r'):
            video_processor.reset_background_subtractor()
            print("Background subtractor reset")
    
    # Calculate and display final FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 0 and frame_count > 0:
        fps = frame_count / elapsed_time
        print(f"\nAverage FPS: {fps:.2f}")
    
    # Release video and close windows
    video_processor.release_video()
    cv2.destroyAllWindows()
    print("\nVideo processing completed.")

if __name__ == "__main__":
    main()