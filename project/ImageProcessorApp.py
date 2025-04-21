import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os

from AerialImageLoader import AerialImageLoader
from VideoProcessor import VideoProcessor

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Video Processor")
        self.image_loader = AerialImageLoader()
        self.video_processor = VideoProcessor()
        self.image_paths = []
        self.video_paths = []
        self.current_image = None
        self.video_playing = False
        self.video_thread = None
        self.selected_tab = "image"
        
        # Configure the root window to handle window closure
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create frames for tabs
        self.image_frame = ttk.Frame(self.notebook, padding="10")
        self.video_frame = ttk.Frame(self.notebook, padding="10")
        
        # Add tabs to notebook
        self.notebook.add(self.image_frame, text="Image Processing")
        self.notebook.add(self.video_frame, text="Video Processing")
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Create widgets for both tabs
        self.create_image_widgets()
        self.create_video_widgets()

    def create_image_widgets(self):
        self.load_img_button = ttk.Button(self.image_frame, text="Load Images", command=self.load_images)
        self.load_img_button.grid(row=0, column=0, padx=5, pady=5)

        self.image_listbox = tk.Listbox(self.image_frame, height=10, width=50)
        self.image_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.action_combobox = ttk.Combobox(self.image_frame, width=30, values=[
            "Denoise", "Sharpen", "Threshold Segmentation", "Otsu Segmentation", 
            "Watershed Segmentation", "GrabCut Segmentation", "Detect Contours", 
            "Detect Features (SIFT)", "Detect Features (ORB)", "Detect Features (HOG)"
        ])
        self.action_combobox.grid(row=2, column=0, padx=5, pady=5)
        self.action_combobox.set("Select Action")

        self.process_img_button = ttk.Button(self.image_frame, text="Process", command=self.process_image)
        self.process_img_button.grid(row=2, column=1, padx=5, pady=5)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def create_video_widgets(self):
        self.load_video_button = ttk.Button(self.video_frame, text="Load Video", command=self.load_video)
        self.load_video_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.open_camera_button = ttk.Button(self.video_frame, text="Open Camera", command=self.open_camera)
        self.open_camera_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.video_listbox = tk.Listbox(self.video_frame, height=5, width=50)
        self.video_listbox.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        
        self.video_action_combobox = ttk.Combobox(self.video_frame, width=30, values=[
            "Normal Playback", "Optical Flow (Farneback)", "Optical Flow (Lucas-Kanade)", 
            "Background Subtraction"
        ])
        self.video_action_combobox.grid(row=2, column=0, padx=5, pady=5)
        self.video_action_combobox.set("Normal Playback")
        self.video_action_combobox.bind('<<ComboboxSelected>>', self.on_video_action_changed)
        
        self.controls_frame = ttk.Frame(self.video_frame)
        self.controls_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        
        self.play_button = ttk.Button(self.controls_frame, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.controls_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(self.controls_frame, text="Reset", command=self.reset_video)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Create a parameters frame that will be shown/hidden based on selected action
        self.params_frame = ttk.LabelFrame(self.video_frame, text="Parameters")
        self.params_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        self.params_frame.grid_remove()  # Initially hidden
        
        # Background Subtraction parameters
        self.bg_params_frame = ttk.Frame(self.params_frame)
        
        # Learning rate slider
        ttk.Label(self.bg_params_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.learning_rate_var = tk.DoubleVar(value=0.002)
        self.learning_rate_slider = ttk.Scale(
            self.bg_params_frame, 
            from_=0.0001, 
            to=0.01, 
            orient="horizontal", 
            length=200, 
            variable=self.learning_rate_var
        )
        self.learning_rate_slider.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.bg_params_frame, textvariable=self.learning_rate_var).grid(row=0, column=2, padx=5, pady=2)
        
        # Min area slider
        ttk.Label(self.bg_params_frame, text="Min Area:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.min_area_var = tk.IntVar(value=300)
        self.min_area_slider = ttk.Scale(
            self.bg_params_frame, 
            from_=50, 
            to=2000, 
            orient="horizontal", 
            length=200, 
            variable=self.min_area_var
        )
        self.min_area_slider.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(self.bg_params_frame, textvariable=self.min_area_var).grid(row=1, column=2, padx=5, pady=2)
        
        # Min speed slider
        ttk.Label(self.bg_params_frame, text="Min Speed:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.min_speed_var = tk.DoubleVar(value=1.5)
        self.min_speed_slider = ttk.Scale(
            self.bg_params_frame, 
            from_=0.5, 
            to=5.0, 
            orient="horizontal", 
            length=200, 
            variable=self.min_speed_var
        )
        self.min_speed_slider.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(self.bg_params_frame, textvariable=self.min_speed_var).grid(row=2, column=2, padx=5, pady=2)
        
        # Apply button
        self.apply_params_button = ttk.Button(self.bg_params_frame, text="Apply", command=self.reset_video)
        self.apply_params_button.grid(row=3, column=1, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        # Label for video info
        self.video_info_label = ttk.Label(self.video_frame, text="No video loaded")
        self.video_info_label.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
        # FPS counter label
        self.fps_label = ttk.Label(self.video_frame, text="FPS: 0.0")
        self.fps_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

    def on_tab_changed(self, event):
        selected_tab = self.notebook.index(self.notebook.select())
        
        # Stop any running video when switching tabs
        if self.selected_tab == "video" and selected_tab == 0:  # Switching from video to image
            self.stop_video()
            
        # Update selected tab state
        self.selected_tab = "image" if selected_tab == 0 else "video"

    def load_images(self):
        self.image_paths = filedialog.askopenfilenames(
            title="Select Images", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, os.path.basename(path))

    def load_video(self):
        video_paths = filedialog.askopenfilenames(
            title="Select Video", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if not video_paths:
            return
            
        self.video_listbox.delete(0, tk.END)
        self.video_paths = list(video_paths)
        
        for path in self.video_paths:
            self.video_listbox.insert(tk.END, os.path.basename(path))

    def open_camera(self):
        # Stop any running video
        self.stop_video()
        
        # Clear existing entries and add camera entry
        self.video_listbox.delete(0, tk.END)
        self.video_listbox.insert(tk.END, "Camera (0)")
        self.video_paths = [0]  # 0 is the index for default camera
        
        # Select the camera in the listbox
        self.video_listbox.selection_set(0)
        self.on_video_select(None)

    def on_image_select(self, event):
        selected_indices = self.image_listbox.curselection()
        if not selected_indices:
            return
            
        selected_index = selected_indices[0]
        if selected_index < len(self.image_paths):
            image_path = self.image_paths[selected_index]
            self.current_image = self.image_loader.load_image(image_path)
            self.display_image(self.current_image, self.image_label)

    def on_video_select(self, event):
        selected_indices = self.video_listbox.curselection()
        if not selected_indices:
            return
            
        # Stop any running video
        self.stop_video()
        
        selected_index = selected_indices[0]
        if selected_index < len(self.video_paths):
            # Try to open the video
            video_path = self.video_paths[selected_index]
            success = self.video_processor.open_video(video_path)
            
            if success:
                # Get and display first frame
                frame = self.video_processor.get_frame()
                if frame is not None:
                    self.display_image(frame, self.video_label)
                
                # Display video properties
                props = self.video_processor.get_video_properties()
                info_text = f"Video: {props['Width']}x{props['Height']} @ {props['FPS']:.2f} FPS"
                self.video_info_label.config(text=info_text)
            else:
                messagebox.showerror("Error", "Failed to open video")

    def display_image(self, image, label_widget):
        # Resize image to fit display (max 800x600)
        h, w = image.shape[:2]
        max_width = 800
        max_height = 600
        
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage for Tkinter
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update label
        label_widget.config(image=image_tk)
        label_widget.image = image_tk  # Keep a reference to prevent garbage collection

    def process_image(self):
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return

        action = self.action_combobox.get()
        if action == "Denoise":
            result_image = self.image_loader.denoise_image(method='gaussian', ksize=(5, 5), sigma=1)
        elif action == "Sharpen":
            result_image = self.image_loader.sharpen_image(method='unsharp_mask', amount=1.5)
        elif action == "Threshold Segmentation":
            result_image = self.image_loader.threshold_segmentation(threshold_value=127)
        elif action == "Otsu Segmentation":
            result_image = self.image_loader.otsu_segmentation()
        elif action == "Watershed Segmentation":
            result_image = self.image_loader.watershed_segmentation()
        elif action == "GrabCut Segmentation":
            rect = (50, 50, 450, 290)  # Example rectangle
            result_image = self.image_loader.grabcut_segmentation(rect)
        elif action == "Detect Contours":
            contours = self.image_loader.detect_contours()
            result_image = self.image_loader.draw_contours(contours)
        elif action == "Detect Features (SIFT)":
            _, result_image = self.image_loader.detect_features(method='SIFT')
        elif action == "Detect Features (ORB)":
            _, result_image = self.image_loader.detect_features(method='ORB')
        elif action == "Detect Features (HOG)":
            _, result_image = self.image_loader.detect_features(method='HOG')
        else:
            messagebox.showerror("Error", "Invalid action selected")
            return

        self.display_image(result_image, self.image_label)

    def play_video(self):
        if not self.video_playing and self.video_processor.video_capture is not None:
            self.video_playing = True
            self.video_thread = threading.Thread(target=self.video_playback_thread)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_video(self):
        self.video_playing = False
        if self.video_thread is not None:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
    
    def reset_video(self):
        # Stop any running video
        self.stop_video()
        
        # Re-select the current video
        selected_indices = self.video_listbox.curselection()
        if selected_indices:
            selected_index = selected_indices[0]
            self.video_listbox.selection_clear(0, tk.END)
            self.video_listbox.selection_set(selected_index)
            self.on_video_select(None)
        
        # Reset background subtractor if needed
        self.video_processor.reset_background_subtractor()
    
    def video_playback_thread(self):
        prev_frame = None
        
        action = self.video_action_combobox.get()
        skip_frames = 0  # For frame skipping to improve performance
        
        # Adjust video capture properties for better performance
        if self.video_processor.video_capture is not None:
            # Try to set capture buffer size (may not work on all platforms)
            self.video_processor.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Get FPS for adaptive timing
        try:
            fps = self.video_processor.get_video_properties()["FPS"]
            if fps <= 0 or fps > 120:  # Default if invalid FPS
                fps = 30
        except:
            fps = 30
            
        # Calculate target frame time
        target_frame_time = 1.0 / fps
        
        while self.video_playing:
            start_time = time.time()  # Track processing time
            
            # Skip frames for heavy processing methods to maintain higher overall FPS
            if action in ["Optical Flow (Farneback)", "Background Subtraction"] and skip_frames < 1:
                skip_frames += 1
                _ = self.video_processor.get_frame()  # Skip this frame
                continue
            else:
                skip_frames = 0
            
            # Get frame
            frame = self.video_processor.get_frame()
            
            if frame is None:
                self.video_playing = False
                break
                
            # Optionally resize the frame for faster processing (especially for Background Subtraction)
            if action == "Background Subtraction":
                # Get original dimensions
                h, w = frame.shape[:2]
                # Keep aspect ratio but reduce size
                scale = 0.75  # Reduce to 75% for background subtraction
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            # Process frame according to selected action
            if action == "Optical Flow (Farneback)" and prev_frame is not None:
                processed_frame = self.video_processor.apply_optical_flow(prev_frame, frame)
                if processed_frame is not None:
                    frame_to_display = processed_frame
                else:
                    frame_to_display = frame
            elif action == "Optical Flow (Lucas-Kanade)" and prev_frame is not None:
                processed_frame = self.video_processor.apply_lucas_kanade_optical_flow(prev_frame, frame)
                if processed_frame is not None:
                    frame_to_display = processed_frame
                else:
                    frame_to_display = frame
            elif action == "Background Subtraction":
                # Use optimized parameters
                learning_rate = 0.002  # Small learning rate for stability
                min_area = 300 * scale**2  # Adjusted for scaled frame
                min_speed = 1.5  # Minimum movement speed to detect
                
                _, processed_frame = self.video_processor.apply_background_subtraction(
                    frame, 
                    learning_rate=learning_rate,
                    min_area=min_area,
                    min_speed=min_speed
                )
                if processed_frame is not None:
                    frame_to_display = processed_frame
                else:
                    frame_to_display = frame
            else:
                frame_to_display = frame
            
            # Save current frame as previous for next iteration (create a copy to avoid reference issues)
            if prev_frame is None or not np.array_equal(prev_frame.shape, frame.shape):
                prev_frame = frame.copy()
            else:
                np.copyto(prev_frame, frame)  # More efficient than creating a new copy
            
            # Update UI with the processed frame
            self.root.after(0, lambda f=frame_to_display: self.display_image(f, self.video_label))
            
            # Calculate elapsed time and sleep as needed to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, target_frame_time - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def on_closing(self):
        # Stop video playback and release resources
        self.stop_video()
        self.video_processor.release_video()
        
        # Destroy root window
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()