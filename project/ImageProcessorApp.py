import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import numpy as np

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
        self.original_image = None
        self.video_playing = False
        self.video_thread = None
        self.selected_tab = "image"
        
        # For FPS tracking
        self.frame_count = 0
        self.fps_update_interval = 1.0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # For perspective transform
        self.perspective_points = []
        self.is_selecting_points = False
        
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
        # Left panel for controls
        left_panel = ttk.Frame(self.image_frame)
        left_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=5, pady=5)
        
        # Right panel for image display
        right_panel = ttk.Frame(self.image_frame)
        right_panel.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(2, weight=1)
        
        # Load button
        self.load_button = ttk.Button(left_panel, text="Load Images", command=self.load_images)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Image listbox
        self.image_listbox = tk.Listbox(left_panel, height=10, width=30)
        self.image_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Scrollbar for listbox
        listbox_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=self.image_listbox.yview)
        listbox_scrollbar.grid(row=1, column=1, pady=5, sticky="ns")
        self.image_listbox.config(yscrollcommand=listbox_scrollbar.set)
        
        # Create processing section with tabs
        processing_tabs = ttk.Notebook(left_panel)
        processing_tabs.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Tab for segmentation methods
        segmentation_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(segmentation_tab, text="Segmentation")
        
        # Tab for feature detection
        feature_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(feature_tab, text="Features")
        
        # Tab for geometric transformations (Week 7)
        geometric_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(geometric_tab, text="Geometric")
        
        # Tab for morphological operations (Week 8)
        morphology_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(morphology_tab, text="Morphology")
        
        # Segmentation controls
        segmentation_methods = ["Denoise", "Sharpen", "Threshold Segmentation", 
                              "Otsu Segmentation", "Watershed Segmentation", "GrabCut Segmentation"]
        
        self.segmentation_combobox = ttk.Combobox(segmentation_tab, values=segmentation_methods)
        self.segmentation_combobox.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.segmentation_combobox.set("Select Segmentation Method")
        
        segmentation_btn = ttk.Button(segmentation_tab, text="Apply", 
                                     command=lambda: self.process_image('segmentation'))
        segmentation_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Feature detection controls
        feature_methods = ["Detect Contours", "Detect Features (SIFT)", 
                          "Detect Features (ORB)", "Detect Features (HOG)"]
        
        self.feature_combobox = ttk.Combobox(feature_tab, values=feature_methods)
        self.feature_combobox.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.feature_combobox.set("Select Feature Method")
        
        feature_btn = ttk.Button(feature_tab, text="Apply", 
                               command=lambda: self.process_image('feature'))
        feature_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Geometric transformation controls (Week 7)
        geometric_methods = ["Resize", "Rotate", "Affine Transform", 
                           "Perspective Transform", "Crop", "Flip"]
        
        self.geometric_combobox = ttk.Combobox(geometric_tab, values=geometric_methods)
        self.geometric_combobox.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.geometric_combobox.set("Select Geometric Transform")
        
        geometric_btn = ttk.Button(geometric_tab, text="Apply", 
                                 command=lambda: self.process_image('geometric'))
        geometric_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Morphological operations controls (Week 8)
        morphology_methods = ["Erosion", "Dilation", "Opening", "Closing", 
                             "Morphological Gradient", "Top Hat", "Black Hat", 
                             "Enhanced Segmentation", "Remove Noise", "Extract Boundaries", 
                             "Skeletonize"]
        
        self.morphology_combobox = ttk.Combobox(morphology_tab, values=morphology_methods)
        self.morphology_combobox.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.morphology_combobox.set("Select Morphological Operation")
        
        # Kernel size slider for morphological operations
        ttk.Label(morphology_tab, text="Kernel Size:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            morphology_tab, 
            from_=1, 
            to=21, 
            orient="horizontal", 
            variable=self.kernel_size_var,
            command=lambda val: self.kernel_size_var.set(int(float(val)) if int(float(val)) % 2 != 0 else int(float(val)) + 1)
        )
        kernel_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(morphology_tab, textvariable=self.kernel_size_var).grid(row=1, column=2, padx=5, pady=2)
        
        # Iterations slider for morphological operations
        ttk.Label(morphology_tab, text="Iterations:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.iterations_var = tk.IntVar(value=1)
        iterations_slider = ttk.Scale(
            morphology_tab, 
            from_=1, 
            to=10, 
            orient="horizontal", 
            variable=self.iterations_var
        )
        iterations_slider.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(morphology_tab, textvariable=self.iterations_var).grid(row=2, column=2, padx=5, pady=2)
        
        # Kernel shape selection for morphological operations
        ttk.Label(morphology_tab, text="Kernel Shape:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.kernel_shape_var = tk.StringVar(value="rect")
        kernel_shapes = ["rect", "ellipse", "cross"]
        kernel_shape_combobox = ttk.Combobox(morphology_tab, values=kernel_shapes, textvariable=self.kernel_shape_var, width=10)
        kernel_shape_combobox.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        
        morphology_btn = ttk.Button(morphology_tab, text="Apply", 
                                  command=lambda: self.process_image('morphology'))
        morphology_btn.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        # Reset button
        reset_btn = ttk.Button(left_panel, text="Reset Image", command=self.reset_image)
        reset_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Save button
        save_btn = ttk.Button(left_panel, text="Save Processed Image", command=self.save_image)
        save_btn.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Image display
        self.canvas_frame = ttk.Frame(right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind canvas events for point selection (for perspective transform)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Configure weights
        left_panel.rowconfigure(1, weight=1)
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)

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

    def on_video_action_changed(self, event):
        """Handle change of video processing action."""
        action = self.video_action_combobox.get()
        
        # Show/hide parameter panels based on selected action
        if action == "Background Subtraction":
            # Show the parameter frame with background subtraction controls
            self.params_frame.grid()
            self.bg_params_frame.pack(fill="both", expand=True)
        else:
            # Hide the parameter frame for other actions
            self.params_frame.grid_remove()
            
        # Stop and reset if we're currently playing
        if self.video_playing:
            self.stop_video()
            self.reset_video()

    def load_images(self):
        self.image_paths = filedialog.askopenfilenames(
            title="Select Images", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, os.path.basename(path))

    def on_image_select(self, event):
        selected_indices = self.image_listbox.curselection()
        if not selected_indices:
            return
            
        selected_index = selected_indices[0]
        if selected_index < len(self.image_paths):
            image_path = self.image_paths[selected_index]
            self.original_image = self.image_loader.load_image(image_path)
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            
            # Reset perspective points
            self.perspective_points = []
            self.is_selecting_points = False

    def display_image(self, image):
        """Display an image on the canvas."""
        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Calculate scaling to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not ready yet
            canvas_width = 800
            canvas_height = 600
            
        img_width, img_height = pil_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        # Apply scaling
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width/2, canvas_height/2, anchor=tk.CENTER, image=self.tk_image)
        
        # If we're selecting points for perspective transform, draw existing points
        if self.is_selecting_points:
            for i, point in enumerate(self.perspective_points):
                # Scale points to match displayed image
                x, y = point
                x_scaled = x * scale
                y_scaled = y * scale
                
                # Center the points on the canvas
                x_centered = (canvas_width - new_width) / 2 + x_scaled
                y_centered = (canvas_height - new_height) / 2 + y_scaled
                
                # Draw point
                self.canvas.create_oval(
                    x_centered-5, y_centered-5, 
                    x_centered+5, y_centered+5, 
                    fill='red'
                )
                
                # Draw point number
                self.canvas.create_text(
                    x_centered+10, y_centered-10, 
                    text=str(i+1), 
                    fill='red'
                )

    def on_canvas_click(self, event):
        """Handle canvas clicks for point selection."""
        if not self.is_selecting_points or self.current_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get image dimensions
        img_height, img_width = self.current_image.shape[:2]
        
        # Calculate scaling factor and offsets
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        x_offset = (canvas_width - new_width) / 2
        y_offset = (canvas_height - new_height) / 2
        
        # Convert canvas coordinates to image coordinates
        if (x_offset <= event.x <= x_offset + new_width and 
            y_offset <= event.y <= y_offset + new_height):
            
            # Convert to original image coordinates
            orig_x = int((event.x - x_offset) / scale)
            orig_y = int((event.y - y_offset) / scale)
            
            # Add point
            if len(self.perspective_points) < 4:
                self.perspective_points.append((orig_x, orig_y))
                
                # Redisplay image with points
                self.display_image(self.current_image)
                
                # If we have enough points, process the transform
                if len(self.perspective_points) == 4 and self.is_selecting_points:
                    self.apply_perspective_transform()

    def apply_perspective_transform(self):
        """Apply perspective transform using the selected points."""
        if len(self.perspective_points) != 4:
            messagebox.showerror("Error", "Please select exactly 4 points")
            return
            
        # Get image dimensions
        height, width = self.current_image.shape[:2]
        
        # Define destination points (rectangle)
        # Sort points to ensure consistent order: top-left, top-right, bottom-right, bottom-left
        src_points = np.array(self.perspective_points, dtype=np.float32)
        
        # Calculate destination points based on the dimensions of the selected quadrilateral
        # Find the width and height of the selected quadrilateral
        width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                           ((src_points[1][1] - src_points[0][1]) ** 2))
        width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                              ((src_points[2][1] - src_points[3][1]) ** 2))
        width_max = max(int(width_top), int(width_bottom))
        
        height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + 
                             ((src_points[3][1] - src_points[0][1]) ** 2))
        height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + 
                              ((src_points[2][1] - src_points[1][1]) ** 2))
        height_max = max(int(height_left), int(height_right))
        
        dst_points = np.array([
            [0, 0],               # top-left
            [width_max - 1, 0],   # top-right
            [width_max - 1, height_max - 1],  # bottom-right
            [0, height_max - 1]   # bottom-left
        ], dtype=np.float32)
        
        # Apply perspective transform
        try:
            self.current_image = self.image_loader.perspective_transform(src_points, dst_points)
            self.display_image(self.current_image)
        except Exception as e:
            messagebox.showerror("Error", f"Perspective transform failed: {str(e)}")
        
        # Reset selection state
        self.is_selecting_points = False
        self.perspective_points = []

    def process_image(self, process_type):
        """Process the current image based on the selected method."""
        if self.current_image is None:
            messagebox.showerror("Error", "No image loaded")
            return
            
        if process_type == 'segmentation':
            action = self.segmentation_combobox.get()
            
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
            else:
                messagebox.showerror("Error", "Invalid segmentation method")
                return
                
        elif process_type == 'feature':
            action = self.feature_combobox.get()
            
            if action == "Detect Contours":
                contours = self.image_loader.detect_contours()
                result_image = self.image_loader.draw_contours(contours)
            elif action == "Detect Features (SIFT)":
                _, result_image = self.image_loader.detect_features(method='SIFT')
            elif action == "Detect Features (ORB)":
                _, result_image = self.image_loader.detect_features(method='ORB')
            elif action == "Detect Features (HOG)":
                _, result_image = self.image_loader.detect_features(method='HOG')
            else:
                messagebox.showerror("Error", "Invalid feature method")
                return
                
        elif process_type == 'geometric':
            action = self.geometric_combobox.get()
            
            if action == "Resize":
                scale = simpledialog.askfloat("Scale Factor", "Enter scale factor:", minvalue=0.1, maxvalue=10.0)
                if scale:
                    result_image = self.image_loader.resize_image(scale=scale)
                else:
                    return
                    
            elif action == "Rotate":
                angle = simpledialog.askfloat("Rotation Angle", "Enter rotation angle (degrees):", minvalue=-360, maxvalue=360)
                if angle is not None:
                    result_image = self.image_loader.rotate_image(angle=angle)
                else:
                    return
                    
            elif action == "Affine Transform":
                messagebox.showinfo("Affine Transform", "Click three points on the image to define the source triangle")
                
                # Get three source points from user
                self.is_selecting_points = True
                self.perspective_points = []
                return  # Wait for points to be selected
                
            elif action == "Perspective Transform":
                messagebox.showinfo("Perspective Transform", "Click four points on the image to define the quadrilateral")
                
                # Get four source points from user
                self.is_selecting_points = True
                self.perspective_points = []
                return  # Wait for points to be selected
                
            elif action == "Crop":
                # Open dialog to get crop rectangle
                crop_dialog = CropDialog(self.root, self.current_image)
                if crop_dialog.result:
                    x, y, w, h = crop_dialog.result
                    result_image = self.image_loader.crop_image(x, y, w, h)
                else:
                    return
                    
            elif action == "Flip":
                # Ask for flip direction
                flip_types = {
                    "Horizontal": 1,
                    "Vertical": 0,
                    "Both": -1
                }
                flip_dialog = FlipDialog(self.root)
                if flip_dialog.result:
                    flip_code = flip_types[flip_dialog.result]
                    result_image = self.image_loader.flip_image(flip_code)
                else:
                    return
            else:
                messagebox.showerror("Error", "Invalid geometric transform")
                return
                
        elif process_type == 'morphology':
            action = self.morphology_combobox.get()
            kernel_size = self.kernel_size_var.get()
            iterations = self.iterations_var.get()
            kernel_shape = self.kernel_shape_var.get()
            
            try:
                if action == "Erosion":
                    result_image = self.image_loader.apply_erosion(kernel_size=kernel_size, iterations=iterations)
                    
                elif action == "Dilation":
                    result_image = self.image_loader.apply_dilation(kernel_size=kernel_size, iterations=iterations)
                    
                elif action == "Opening":
                    result_image = self.image_loader.apply_opening(kernel_size=kernel_size, iterations=iterations)
                    
                elif action == "Closing":
                    result_image = self.image_loader.apply_closing(kernel_size=kernel_size, iterations=iterations)
                    
                elif action == "Morphological Gradient":
                    result_image = self.image_loader.apply_morphological_gradient(kernel_size=kernel_size)
                    
                elif action == "Top Hat":
                    result_image = self.image_loader.apply_top_hat(kernel_size=kernel_size)
                    
                elif action == "Black Hat":
                    result_image = self.image_loader.apply_black_hat(kernel_size=kernel_size)
                    
                elif action == "Enhanced Segmentation":
                    # Configure enhanced segmentation
                    segmentation_methods = ["threshold", "otsu", "adaptive"]
                    segmentation_method = simpledialog.askstring(
                        "Segmentation Method", 
                        "Enter segmentation method (threshold, otsu, adaptive):",
                        initialvalue="otsu"
                    )
                    
                    if not segmentation_method or segmentation_method not in segmentation_methods:
                        messagebox.showerror("Error", "Invalid segmentation method")
                        return
                        
                    # Define sequence of morphological operations
                    operations = [
                        {'op': 'open', 'kernel_size': kernel_size, 'iterations': 1, 'kernel_shape': kernel_shape},
                        {'op': 'close', 'kernel_size': kernel_size, 'iterations': 1, 'kernel_shape': kernel_shape}
                    ]
                    
                    result_image = self.image_loader.enhance_segmentation_with_morphology(
                        segmentation_method=segmentation_method,
                        morphology_operations=operations
                    )
                    
                elif action == "Remove Noise":
                    noise_types = ["salt_pepper", "speckle", "small_holes"]
                    noise_type = simpledialog.askstring(
                        "Noise Type",
                        "Enter noise type (salt_pepper, speckle, small_holes):",
                        initialvalue="salt_pepper"
                    )
                    
                    if not noise_type or noise_type not in noise_types:
                        messagebox.showerror("Error", "Invalid noise type")
                        return
                        
                    result_image = self.image_loader.remove_noise_with_morphology(
                        kernel_size=kernel_size,
                        noise_type=noise_type
                    )
                    
                elif action == "Extract Boundaries":
                    result_image = self.image_loader.extract_boundaries_with_morphology(kernel_size=kernel_size)
                    
                elif action == "Skeletonize":
                    result_image = self.image_loader.skeletonize_image()
                    
                else:
                    messagebox.showerror("Error", "Invalid morphological operation")
                    return
                    
            except Exception as e:
                messagebox.showerror("Error", f"Morphological operation failed: {str(e)}")
                return
                
        else:
            messagebox.showerror("Error", "Invalid process type")
            return
            
        # Update and display the result
        self.current_image = result_image
        self.display_image(self.current_image)

    def reset_image(self):
        """Reset to the original image."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            self.perspective_points = []
            self.is_selecting_points = False

    def save_image(self):
        """Save the processed image."""
        if self.current_image is None:
            messagebox.showerror("Error", "No image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_image)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

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
                    self.display_video_frame(frame)
                
                # Display video properties
                props = self.video_processor.get_video_properties()
                info_text = f"Video: {props['Width']}x{props['Height']} @ {props['FPS']:.2f} FPS"
                self.video_info_label.config(text=info_text)
            else:
                messagebox.showerror("Error", "Failed to open video")

    def display_video_frame(self, frame):
        # Resize frame to fit display (max 800x600)
        h, w = frame.shape[:2]
        max_width = 800
        max_height = 600
        
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage for Tkinter
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.video_label.config(image=image_tk)
        self.video_label.image = image_tk  # Keep a reference to prevent garbage collection

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
        
        # Reset FPS counter
        self.frame_count = 0
        self.last_fps_time = time.time()
        
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
                
            # Update FPS counter
            self.frame_count += 1
            elapsed_since_last_fps = time.time() - self.last_fps_time
            if elapsed_since_last_fps >= self.fps_update_interval:
                self.current_fps = self.frame_count / elapsed_since_last_fps
                self.frame_count = 0
                self.last_fps_time = time.time()
                # Update FPS display
                self.root.after(0, lambda fps=self.current_fps: self.fps_label.config(text=f"FPS: {fps:.1f}"))
            
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
                # Use parameters from the UI sliders
                learning_rate = self.learning_rate_var.get()
                min_area = self.min_area_var.get() * (scale ** 2)  # Adjust for scaled frame
                min_speed = self.min_speed_var.get()
                
                try:
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
                except Exception as e:
                    print(f"Error in background subtraction: {e}")
                    frame_to_display = frame
            else:
                frame_to_display = frame
            
            # Save current frame as previous for next iteration
            if prev_frame is None or not np.array_equal(prev_frame.shape, frame.shape):
                prev_frame = frame.copy()
            else:
                np.copyto(prev_frame, frame)  # More efficient than creating a new copy
            
            # Update UI with the processed frame
            self.root.after(0, lambda f=frame_to_display: self.display_video_frame(f))
            
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


class CropDialog(tk.Toplevel):
    """Dialog for selecting crop region."""
    def __init__(self, parent, image):
        super().__init__(parent)
        self.title("Select Crop Region")
        self.result = None
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Frame for input fields
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill="both", expand=True)
        
        # X input
        ttk.Label(frame, text="X:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.x_var = tk.IntVar(value=0)
        x_spinbox = ttk.Spinbox(frame, from_=0, to=w-1, textvariable=self.x_var, width=10)
        x_spinbox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Y input
        ttk.Label(frame, text="Y:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.y_var = tk.IntVar(value=0)
        y_spinbox = ttk.Spinbox(frame, from_=0, to=h-1, textvariable=self.y_var, width=10)
        y_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Width input
        ttk.Label(frame, text="Width:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.width_var = tk.IntVar(value=w)
        width_spinbox = ttk.Spinbox(frame, from_=1, to=w, textvariable=self.width_var, width=10)
        width_spinbox.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Height input
        ttk.Label(frame, text="Height:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.height_var = tk.IntVar(value=h)
        height_spinbox = ttk.Spinbox(frame, from_=1, to=h, textvariable=self.height_var, width=10)
        height_spinbox.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        x = self.x_var.get()
        y = self.y_var.get()
        width = self.width_var.get()
        height = self.height_var.get()
        
        self.result = (x, y, width, height)
        self.destroy()
    
    def on_cancel(self):
        self.destroy()


class FlipDialog(tk.Toplevel):
    """Dialog for selecting flip direction."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Flip Direction")
        self.result = None
        
        # Frame for radio buttons
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill="both", expand=True)
        
        # Radio buttons
        self.direction_var = tk.StringVar(value="Horizontal")
        
        ttk.Radiobutton(frame, text="Horizontal", variable=self.direction_var, 
                      value="Horizontal").pack(anchor="w", padx=5, pady=5)
        ttk.Radiobutton(frame, text="Vertical", variable=self.direction_var, 
                      value="Vertical").pack(anchor="w", padx=5, pady=5)
        ttk.Radiobutton(frame, text="Both (Horizontal & Vertical)", variable=self.direction_var, 
                      value="Both").pack(anchor="w", padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        self.result = self.direction_var.get()
        self.destroy()
    
    def on_cancel(self):
        self.destroy()


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.geometry("1200x800")
    root.mainloop()


if __name__ == "__main__":
    main()