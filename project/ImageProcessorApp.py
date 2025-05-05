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
from ObjectClassifier import ObjectClassifier

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

    def load_pretrained_model(self):
        """Load selected pre-trained model"""
        model_type = self.pretrained_model_var.get()
        
        try:
            if model_type == "ResNet50":
                self.status_var.set("Loading ResNet50...")
                self.root.update()
                self.image_loader.load_resnet()
                self.status_var.set("ResNet50 loaded successfully")
                
            elif model_type == "MobileNetV2":
                self.status_var.set("Loading MobileNetV2...")
                self.root.update()
                self.image_loader.load_mobilenet()
                self.status_var.set("MobileNetV2 loaded successfully")
                
            elif model_type == "YOLOv8":
                yolo_type = self.yolo_type_var.get()
                self.status_var.set(f"Loading {yolo_type}...")
                self.root.update()
                self.image_loader.load_yolo(model_type=yolo_type)
                self.status_var.set(f"{yolo_type} loaded successfully")
                
            elif model_type == "U-Net":
                # Ask if user wants to load existing model or create new
                choice = messagebox.askyesno("U-Net", "Load existing U-Net model?\nYes = Load existing\nNo = Create new")
                if choice:
                    # Load existing model
                    model_path = filedialog.askopenfilename(
                        title="Load U-Net Model",
                        filetypes=[("Keras Model", "*.h5"), ("SavedModel", "*.pb"), ("All Files", "*.*")]
                    )
                    if model_path:
                        self.status_var.set("Loading U-Net model...")
                        self.root.update()
                        self.image_loader.load_pretrained_unet(model_path)
                        self.status_var.set("U-Net loaded successfully")
                else:
                    # Create new model
                    self.status_var.set("Creating new U-Net...")
                    self.root.update()
                    self.image_loader.create_unet()
                    self.status_var.set("U-Net created successfully")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed")

    def browse_test_dataset(self):
        """Browse for test dataset directory"""
        directory = filedialog.askdirectory(title="Select Test Dataset Directory")
        if directory:
            self.test_dataset_path_var.set(directory)

    def browse_test_output_dir(self):
        """Browse for test output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory for Test Results")
        if directory:
            self.test_output_dir_var.set(directory)

    def run_model_tests(self):
        """Run comprehensive model testing"""
        test_dataset_dir = self.test_dataset_path_var.get()
        output_dir = self.test_output_dir_var.get()
        
        if not test_dataset_dir:
            messagebox.showerror("Error", "Please select test dataset directory")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please select output directory")
            return
        
        try:
            # Get selected models
            models_to_test = []
            for model_type, var in self.test_models.items():
                if var.get():
                    models_to_test.append(model_type)
            
            if not models_to_test:
                messagebox.showerror("Error", "Please select at least one model to test")
                return
            
            self.status_var.set("Running model tests...")
            self.root.update()
            
            # Run appropriate testing based on options
            if self.test_with_error_analysis_var.get():
                results, error_analysis = self.image_loader.test_with_error_analysis(
                    test_dataset_dir, output_dir
                )
                self.status_var.set("Testing completed with error analysis")
            else:
                results = self.image_loader.test_model_performance(
                    test_dataset_dir, output_dir, models_to_test
                )
                self.status_var.set("Testing completed")
            
            # Run efficiency comparison if requested
            if self.efficiency_comparison_var.get():
                efficiency_metrics = self.image_loader.compare_model_efficiency(
                    test_dataset_dir, output_dir
                )
            
            # Show summary in dialog
            self.show_test_summary(results)
            
            # Ask if user wants to view detailed results
            if messagebox.askyesno("Test Complete", 
                                f"Testing completed! Results saved to:\n{output_dir}\n\nView detailed results now?"):
                self.view_test_results()
        
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")
            self.status_var.set("Testing failed")

    def show_test_summary(self, results):
        """Show testing summary dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Model Testing Summary")
        dialog.geometry("800x600")
        dialog.resizable(True, True)
        
        # Main frame
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill="both", expand=True)
        
        # Create text area with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True)
        
        text_widget = tk.Text(text_frame, wrap="word", height=25, width=60)
        text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Format summary text
        summary_text = "MODEL TESTING SUMMARY\n"
        summary_text += "=" * 50 + "\n\n"
        
        for model_name, model_results in results.items():
            summary_text += f"{model_name}:\n"
            summary_text += f"  Accuracy: {model_results['accuracy']:.3f}\n"
            summary_text += f"  FPS: {model_results['fps']:.2f}\n"
            summary_text += f"  Test Time: {model_results['test_time']:.2f}s\n"
            if 'average_confidence' in model_results:
                summary_text += f"  Avg Confidence: {model_results['average_confidence']:.3f}\n"
            summary_text += "\n"
        
        # Insert text
        text_widget.insert("1.0", summary_text)
        text_widget.config(state="disabled")
        
        # Close button
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"{width}x{height}+{x}+{y}")

    def view_test_results(self):
        """View detailed test results"""
        output_dir = self.test_output_dir_var.get()
        
        if not output_dir:
            messagebox.showerror("Error", "No test results directory selected")
            return
        
        import os
        import webbrowser
        import platform
        
        # Create HTML report viewer
        self.generate_html_report(output_dir)
        
        # Open the report
        report_path = os.path.join(output_dir, "test_report.html")
        if os.path.exists(report_path):
            # Open in default browser
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
        else:
            # Fallback: open directory
            if platform.system() == 'Darwin':  # macOS
                os.system(f'open "{output_dir}"')
            elif platform.system() == 'Windows':
                os.startfile(output_dir)
            else:  # Linux
                os.system(f'xdg-open "{output_dir}"')

    def generate_html_report(self, output_dir):
        """Generate HTML report from test results"""
        import os
        from pathlib import Path
        
        report_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Testing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .section { margin-bottom: 30px; }
                img { max-width: 100%; height: auto; margin-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .performance { display: flex; flex-wrap: wrap; justify-content: space-around; }
                .performance-item { margin: 10px; }
            </style>
        </head>
        <body>
            <h1>Aerial Object Recognition Model Testing Report</h1>
            
            <div class="section">
                <h2>Performance Overview</h2>
                <img src="performance_comparison.png" alt="Performance Comparison">
            </div>
            
            <div class="section">
                <h2>Timing Analysis</h2>
                <img src="timing_analysis.png" alt="Timing Analysis">
            </div>
            
            <div class="section">
                <h2>Confusion Matrices</h2>
                <img src="confusion_matrices.png" alt="Confusion Matrices">
            </div>
        """
        
        # Add error analysis if available
        if os.path.exists(os.path.join(output_dir, "model_agreement_matrix.png")):
            report_html += """
            <div class="section">
                <h2>Error Analysis</h2>
                <h3>Model Agreement Matrix</h3>
                <img src="model_agreement_matrix.png" alt="Model Agreement Matrix">
                
                <h3>Per-Class Accuracy</h3>
                <img src="per_class_accuracy.png" alt="Per-Class Accuracy">
            </div>
            """
        
        # Add efficiency comparison if available
        if os.path.exists(os.path.join(output_dir, "efficiency_comparison.png")):
            report_html += """
            <div class="section">
                <h2>Efficiency Comparison</h2>
                <img src="efficiency_comparison.png" alt="Efficiency Comparison">
            </div>
            """
        
        report_html += """
        </body>
        </html>
        """
    
        # Save HTML report
        with open(os.path.join(output_dir, "test_report.html"), "w") as f:
            f.write(report_html)

    def run_pretrained_prediction(self):
        """Run prediction with selected pre-trained model"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        model_type = self.pretrained_model_var.get()
        
        try:
            if model_type == "ResNet50":
                self.status_var.set("Running ResNet50 prediction...")
                self.root.update()
                results = self.image_loader.predict_with_resnet()
                
                # Display results
                result_text = "ResNet50 Predictions:\n"
                for i, (class_id, description, score) in enumerate(results):
                    result_text += f"{i+1}. {description} ({score:.3f})\n"
                
                self.show_results_dialog("ResNet50 Results", result_text)
                
            elif model_type == "MobileNetV2":
                self.status_var.set("Running MobileNetV2 prediction...")
                self.root.update()
                results = self.image_loader.predict_with_mobilenet()
                
                # Display results
                result_text = "MobileNetV2 Predictions:\n"
                for i, (class_id, description, score) in enumerate(results):
                    result_text += f"{i+1}. {description} ({score:.3f})\n"
                
                self.show_results_dialog("MobileNetV2 Results", result_text)
                
            elif model_type == "YOLOv8":
                self.status_var.set("Running YOLO detection...")
                self.root.update()
                conf_threshold = self.yolo_conf_var.get()
                detections, annotated_img = self.image_loader.detect_with_yolo(conf_threshold=conf_threshold)
                
                # Update current image with annotations
                self.current_image = annotated_img
                self.display_image(self.current_image)
                
                # Show detection results
                result_text = "YOLO Detections:\n"
                for det in detections:
                    result_text += f"- {det['class']}: {det['confidence']:.3f} @ {det['bbox']}\n"
                
                self.show_results_dialog("YOLO Results", result_text)
                
            elif model_type == "U-Net":
                self.status_var.set("Running U-Net segmentation...")
                self.root.update()
                mask, overlaid = self.image_loader.segment_with_unet()
                
                # Show options for display
                choice = messagebox.askyesno("U-Net Results", "Show overlaid result?\nYes = Show overlay\nNo = Show mask")
                if choice:
                    self.current_image = overlaid
                else:
                    # Convert mask to RGB for display
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    self.current_image = mask_rgb
                
                self.display_image(self.current_image)
                self.status_var.set("U-Net segmentation completed")
                
            self.status_var.set(f"{model_type} prediction completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")

    def compare_pretrained_models(self):
        """Compare results from all pre-trained models"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            self.status_var.set("Comparing models...")
            self.root.update()
            
            results = self.image_loader.compare_pretrained_models()
            
            # Format results for display
            comparison_text = "Model Comparison Results:\n\n"
            
            # ResNet results
            if isinstance(results['resnet'], list):
                comparison_text += "ResNet50:\n"
                for i, (_, description, score) in enumerate(results['resnet'][:3]):
                    comparison_text += f"  {i+1}. {description} ({score:.3f})\n"
            else:
                comparison_text += f"ResNet50: {results['resnet']}\n"
            
            comparison_text += "\n"
            
            # MobileNet results
            if isinstance(results['mobilenet'], list):
                comparison_text += "MobileNetV2:\n"
                for i, (_, description, score) in enumerate(results['mobilenet'][:3]):
                    comparison_text += f"  {i+1}. {description} ({score:.3f})\n"
            else:
                comparison_text += f"MobileNetV2: {results['mobilenet']}\n"
            
            comparison_text += "\n"
            
            # YOLO results
            if isinstance(results['yolo'], list):
                comparison_text += "YOLO:\n"
                for det in results['yolo'][:5]:
                    comparison_text += f"  - {det['class']}: {det['confidence']:.3f}\n"
            else:
                comparison_text += f"YOLO: {results['yolo']}\n"
            
            self.show_results_dialog("Model Comparison", comparison_text)
            self.status_var.set("Model comparison completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
            self.status_var.set("Comparison failed")

    def show_pretrained_model_info(self):
        """Show information about selected pre-trained model"""
        model_type = self.pretrained_model_var.get()
        
        try:
            model_name_map = {
                "ResNet50": "resnet",
                "MobileNetV2": "mobilenet",
                "YOLOv8": "yolo",
                "U-Net": "unet"
            }
            
            model_name = model_name_map.get(model_type)
            if model_name:
                info = self.image_loader.get_pretrained_model_info(model_name)
                self.show_results_dialog(f"{model_type} Information", info)
            else:
                messagebox.showinfo("Info", f"Unknown model type: {model_type}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get model info: {str(e)}")

    def apply_individual_augmentation(self, augmentation_type):
        """Apply individual augmentation to current image"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            if augmentation_type == 'brightness':
                factor = self.brightness_var.get()
                result = self.image_loader.apply_specific_augmentation(
                    'brightness', brightness_range=(factor, factor))
            
            elif augmentation_type == 'contrast':
                factor = self.contrast_var.get()
                result = self.image_loader.apply_specific_augmentation(
                    'contrast', contrast_range=(factor, factor))
            
            elif augmentation_type == 'noise':
                noise_type = self.noise_type_var.get()
                if noise_type == 'gaussian':
                    result = self.image_loader.apply_specific_augmentation('gaussian_noise')
                else:
                    result = self.image_loader.apply_specific_augmentation('salt_pepper_noise')
            
            elif augmentation_type == 'blur':
                blur_type = self.blur_type_var.get()
                result = self.image_loader.apply_specific_augmentation('blur', blur_type=blur_type)
            
            elif augmentation_type == 'color':
                result = self.image_loader.apply_specific_augmentation('color')
            
            self.current_image = result
            self.display_image(self.current_image)
            self.status_var.set(f"Applied {augmentation_type} augmentation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Augmentation failed: {str(e)}")
            self.status_var.set("Augmentation failed")

    def apply_random_augmentation(self):
        """Apply random augmentation with selected parameters"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            # Get selected augmentation parameters
            params = {key: var.get() for key, var in self.aug_params.items()}
            
            # Apply augmentation
            result = self.image_loader.augment_loaded_image(params)
            
            self.current_image = result
            self.display_image(self.current_image)
            self.status_var.set("Applied random augmentation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Random augmentation failed: {str(e)}")
            self.status_var.set("Random augmentation failed")

    def visualize_augmentations(self):
        """Visualize multiple augmentation samples"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            # Get selected augmentation parameters
            params = {key: var.get() for key, var in self.aug_params.items()}
            
            # Visualize augmentations
            fig = self.image_loader.visualize_augmentations(num_samples=8, augmentation_params=params)
            
            # Display the figure
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.show()
            
            self.status_var.set("Visualized augmentations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
            self.status_var.set("Visualization failed")

    def compare_augmentations(self):
        """Compare different augmentation effects"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            # Compare different augmentation types
            fig = self.image_loader.compare_augmentation_effects()
            
            # Display the figure
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.show()
            
            self.status_var.set("Compared augmentation effects")
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
            self.status_var.set("Comparison failed")

    def preview_augmentation_pipeline(self):
        """Preview augmentation pipeline effects"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            severity = self.pipeline_severity_var.get()
            
            # Preview pipeline
            fig = self.image_loader.preview_augmentation_pipeline(severity)
            
            # Display the figure
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.show()
            
            self.status_var.set(f"Previewed {severity} augmentation pipeline")
            
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline preview failed: {str(e)}")
            self.status_var.set("Pipeline preview failed")

    def augment_dataset(self):
        """Augment an entire dataset"""
        # Get input directory
        input_dir = filedialog.askdirectory(title="Select Input Dataset Directory")
        if not input_dir:
            return
        
        # Get output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory for Augmented Dataset")
        if not output_dir:
            return
        
        try:
            # Get augmentation parameters
            params = {key: var.get() for key, var in self.aug_params.items()}
            factor = self.aug_factor_var.get()
            
            self.status_var.set(f"Augmenting dataset (factor: {factor})...")
            self.root.update()
            
            # Augment dataset
            stats = self.image_loader.augment_dataset(
                input_dir,
                output_dir,
                augmentation_factor=factor,
                augmentation_params=params
            )
            
            # Show results
            result_text = f"Dataset Augmentation Results:\n\n"
            result_text += f"Original images: {stats['original_images']}\n"
            result_text += f"Augmented images: {stats['augmented_images']}\n"
            result_text += f"Failed images: {stats['failed_images']}\n"
            result_text += f"Total output images: {stats['original_images'] + stats['augmented_images']}"
            
            self.show_results_dialog("Dataset Augmentation Complete", result_text)
            self.status_var.set("Dataset augmentation completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Dataset augmentation failed: {str(e)}")
            self.status_var.set("Dataset augmentation failed")

    def train_with_augmentation(self):
        """Train model with augmented data"""
        # Get dataset directory
        dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if not dataset_dir:
            return
        
        try:
            # Get parameters
            params = {key: var.get() for key, var in self.aug_params.items()}
            factor = self.aug_factor_var.get()
            model_type = self.classifier_type_var.get()
            
            self.status_var.set(f"Training with augmentation (factor: {factor})...")
            self.root.update()
            
            # Train with augmentation
            if model_type == "CNN":
                epochs = self.epochs_var.get()
                batch_size = self.batch_size_var.get()
                
                result = self.image_loader.train_with_augmentation(
                    dataset_dir,
                    augmentation_factor=factor,
                    model_type='CNN',
                    epochs=epochs,
                    batch_size=batch_size,
                    augmentation_params=params
                )
                
                # Show training results
                final_accuracy = result.history['accuracy'][-1]
                final_val_accuracy = result.history['val_accuracy'][-1]
                
                result_text = f"Training Results:\n\n"
                result_text += f"Final Accuracy: {final_accuracy:.3f}\n"
                result_text += f"Final Validation Accuracy: {final_val_accuracy:.3f}\n"
                result_text += f"Epochs: {epochs}\n"
                result_text += f"Augmentation Factor: {factor}"
                
            else:
                accuracy, report = self.image_loader.train_with_augmentation(
                    dataset_dir,
                    augmentation_factor=factor,
                    model_type=model_type,
                    augmentation_params=params
                )
                
                result_text = f"Training Results:\n\n"
                result_text += f"Accuracy: {accuracy:.3f}\n"
                result_text += f"Augmentation Factor: {factor}\n\n"
                result_text += "Classification Report:\n"
                result_text += str(report)
            
            self.show_results_dialog("Training Complete", result_text)
            self.status_var.set("Training with augmentation completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training with augmentation failed: {str(e)}")
            self.status_var.set("Training with augmentation failed")

    def show_results_dialog(self, title, content):
        """Show results in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("600x400")
        dialog.resizable(True, True)
        
        # Main frame
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill="both", expand=True)
        
        # Text widget
        text_widget = tk.Text(frame, wrap="word", width=60, height=20)
        text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert content
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")
        
        # Close button
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"{width}x{height}+{x}+{y}")

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
        
        # Tab for object classification (Week 9 & 10)
        classification_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(classification_tab, text="Classification")
        
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
        
        # Classification controls (Week 9 & 10)
        # Model selection frame
        model_frame = ttk.LabelFrame(classification_tab, text="Model Selection")
        model_frame.pack(fill="x", expand=False, pady=5)
        
        # Traditional classifier controls
        ttk.Label(model_frame, text="Traditional Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value="SVM")
        model_types = ["SVM", "RandomForest", "KNN"]
        model_combobox = ttk.Combobox(model_frame, values=model_types, textvariable=self.model_var, width=15)
        model_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Feature extractor selection
        ttk.Label(model_frame, text="Feature Extractor:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.feature_extractor_var = tk.StringVar(value="HOG")
        feature_extractors = ["HOG", "SIFT", "ORB"]
        feature_extractor_combobox = ttk.Combobox(model_frame, values=feature_extractors, textvariable=self.feature_extractor_var, width=15)
        feature_extractor_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Model type selection (Traditional vs CNN)
        model_type_frame = ttk.LabelFrame(classification_tab, text="Classifier Type")
        model_type_frame.pack(fill="x", expand=False, pady=5)
        
        self.classifier_type_var = tk.StringVar(value="Traditional")
        ttk.Radiobutton(model_type_frame, text="Traditional ML", variable=self.classifier_type_var, value="Traditional").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(model_type_frame, text="CNN Deep Learning", variable=self.classifier_type_var, value="CNN").pack(anchor="w", padx=5, pady=2)
        
        # CNN parameters frame
        cnn_params_frame = ttk.LabelFrame(classification_tab, text="CNN Parameters")
        cnn_params_frame.pack(fill="x", expand=False, pady=5)
        
        ttk.Label(cnn_params_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_var = tk.IntVar(value=50)
        epochs_spinbox = ttk.Spinbox(cnn_params_frame, from_=10, to=200, textvariable=self.epochs_var, width=10)
        epochs_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(cnn_params_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.batch_size_var = tk.IntVar(value=32)
        batch_sizes = [16, 32, 64, 128]
        batch_size_combobox = ttk.Combobox(cnn_params_frame, values=batch_sizes, textvariable=self.batch_size_var, width=10)
        batch_size_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Training buttons
        train_frame = ttk.Frame(classification_tab)
        train_frame.pack(fill="x", expand=False, pady=5)
        
        train_btn = ttk.Button(train_frame, text="Train Model", command=self.train_classifier)
        train_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        classify_btn = ttk.Button(train_frame, text="Classify Image", command=self.classify_image)
        classify_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Training history button for CNN
        history_btn = ttk.Button(train_frame, text="Show Training History", command=self.show_training_history)
        history_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model summary button for CNN
        summary_btn = ttk.Button(train_frame, text="Show Model Summary", command=self.show_model_summary)
        summary_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Load/save model buttons
        model_file_frame = ttk.Frame(classification_tab)
        model_file_frame.pack(fill="x", expand=False, pady=5)
        
        load_model_btn = ttk.Button(model_file_frame, text="Load Model", command=self.load_classifier_model)
        load_model_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        save_model_btn = ttk.Button(model_file_frame, text="Save Model", command=self.save_classifier_model)
        save_model_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Tab for pre-trained models (Week 11)
        pretrained_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(pretrained_tab, text="Pre-trained Models")

        # Model selection frame
        model_selection_frame = ttk.LabelFrame(pretrained_tab, text="Model Selection")
        model_selection_frame.pack(fill="x", expand=False, pady=5)

        # Model type selection
        ttk.Label(model_selection_frame, text="Model Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pretrained_model_var = tk.StringVar(value="ResNet50")
        pretrained_models = ["ResNet50", "MobileNetV2", "YOLOv8", "U-Net"]
        pretrained_model_combo = ttk.Combobox(model_selection_frame, values=pretrained_models, textvariable=self.pretrained_model_var, width=15)
        pretrained_model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # YOLO model type (only for YOLO)
        ttk.Label(model_selection_frame, text="YOLO Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.yolo_type_var = tk.StringVar(value="yolov8n")
        yolo_types = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        yolo_type_combo = ttk.Combobox(model_selection_frame, values=yolo_types, textvariable=self.yolo_type_var, width=15)
        yolo_type_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # YOLO confidence threshold
        ttk.Label(model_selection_frame, text="YOLO Confidence:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.yolo_conf_var = tk.DoubleVar(value=0.5)
        yolo_conf_slider = ttk.Scale(
            model_selection_frame,
            from_=0.1,
            to=0.9,
            orient="horizontal",
            variable=self.yolo_conf_var
        )
        yolo_conf_slider.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(model_selection_frame, textvariable=self.yolo_conf_var).grid(row=2, column=2, padx=5, pady=5)

        # Action buttons
        action_frame = ttk.Frame(pretrained_tab)
        action_frame.pack(fill="x", expand=False, pady=5)

        load_pretrained_btn = ttk.Button(action_frame, text="Load Model", command=self.load_pretrained_model)
        load_pretrained_btn.pack(side=tk.LEFT, padx=5, pady=5)

        predict_pretrained_btn = ttk.Button(action_frame, text="Run Prediction", command=self.run_pretrained_prediction)
        predict_pretrained_btn.pack(side=tk.LEFT, padx=5, pady=5)

        compare_models_btn = ttk.Button(action_frame, text="Compare Models", command=self.compare_pretrained_models)
        compare_models_btn.pack(side=tk.LEFT, padx=5, pady=5)

        show_model_info_btn = ttk.Button(action_frame, text="Show Model Info", command=self.show_pretrained_model_info)
        show_model_info_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Tab for data augmentation (Week 12)
        augmentation_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(augmentation_tab, text="Data Augmentation")

        # Individual augmentation controls
        individual_frame = ttk.LabelFrame(augmentation_tab, text="Individual Augmentations")
        individual_frame.pack(fill="x", expand=False, pady=5)

        # Brightness controls
        brightness_frame = ttk.Frame(individual_frame)
        brightness_frame.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_slider = ttk.Scale(brightness_frame, from_=0.5, to=1.5, orient="horizontal", 
                                    variable=self.brightness_var, length=150)
        brightness_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(brightness_frame, textvariable=self.brightness_var).pack(side=tk.LEFT, padx=5)

        # Contrast controls
        contrast_frame = ttk.Frame(individual_frame)
        contrast_frame.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=1.5, orient="horizontal", 
                                variable=self.contrast_var, length=150)
        contrast_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(contrast_frame, textvariable=self.contrast_var).pack(side=tk.LEFT, padx=5)

        # Noise controls
        noise_frame = ttk.Frame(individual_frame)
        noise_frame.grid(row=2, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(noise_frame, text="Noise:").pack(side=tk.LEFT, padx=5)
        self.noise_type_var = tk.StringVar(value="gaussian")
        noise_types = ["gaussian", "salt_pepper"]
        noise_type_combo = ttk.Combobox(noise_frame, values=noise_types, textvariable=self.noise_type_var, width=15)
        noise_type_combo.pack(side=tk.LEFT, padx=5)

        # Blur controls
        blur_frame = ttk.Frame(individual_frame)
        blur_frame.grid(row=3, column=0, padx=5, pady=2, sticky="ew")
        ttk.Label(blur_frame, text="Blur:").pack(side=tk.LEFT, padx=5)
        self.blur_type_var = tk.StringVar(value="gaussian")
        blur_types = ["gaussian", "motion", "average"]
        blur_type_combo = ttk.Combobox(blur_frame, values=blur_types, textvariable=self.blur_type_var, width=15)
        blur_type_combo.pack(side=tk.LEFT, padx=5)

        # Individual augmentation buttons
        aug_buttons_frame = ttk.Frame(individual_frame)
        aug_buttons_frame.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        apply_brightness_btn = ttk.Button(aug_buttons_frame, text="Apply Brightness", 
                                        command=lambda: self.apply_individual_augmentation('brightness'))
        apply_brightness_btn.pack(side=tk.LEFT, padx=5)

        apply_contrast_btn = ttk.Button(aug_buttons_frame, text="Apply Contrast", 
                                    command=lambda: self.apply_individual_augmentation('contrast'))
        apply_contrast_btn.pack(side=tk.LEFT, padx=5)

        apply_noise_btn = ttk.Button(aug_buttons_frame, text="Add Noise", 
                                    command=lambda: self.apply_individual_augmentation('noise'))
        apply_noise_btn.pack(side=tk.LEFT, padx=5)

        apply_blur_btn = ttk.Button(aug_buttons_frame, text="Apply Blur", 
                                command=lambda: self.apply_individual_augmentation('blur'))
        apply_blur_btn.pack(side=tk.LEFT, padx=5)

        apply_color_btn = ttk.Button(aug_buttons_frame, text="Apply Color Aug", 
                                    command=lambda: self.apply_individual_augmentation('color'))
        apply_color_btn.pack(side=tk.LEFT, padx=5)

        # Random augmentation controls
        random_frame = ttk.LabelFrame(augmentation_tab, text="Random Augmentation")
        random_frame.pack(fill="x", expand=False, pady=5)

        # Augmentation parameters checkboxes
        params_grid = ttk.Frame(random_frame)
        params_grid.pack(fill="x", padx=5, pady=5)

        self.aug_params = {
            'geometry': tk.BooleanVar(value=True),
            'brightness': tk.BooleanVar(value=True),
            'contrast': tk.BooleanVar(value=True),
            'noise': tk.BooleanVar(value=True),
            'blur': tk.BooleanVar(value=True),
            'color': tk.BooleanVar(value=True),
            'elastic': tk.BooleanVar(value=False)
        }

        row, col = 0, 0
        for param, var in self.aug_params.items():
            cb = ttk.Checkbutton(params_grid, text=param.replace('_', ' ').title(), variable=var)
            cb.grid(row=row, column=col, padx=5, pady=2, sticky="w")
            col += 1
            if col > 2:
                col = 0
                row += 1

        # Random augmentation buttons
        random_buttons_frame = ttk.Frame(random_frame)
        random_buttons_frame.pack(fill="x", padx=5, pady=5)

        apply_random_btn = ttk.Button(random_buttons_frame, text="Apply Random Augmentation", 
                                    command=self.apply_random_augmentation)
        apply_random_btn.pack(side=tk.LEFT, padx=5)

        visualize_aug_btn = ttk.Button(random_buttons_frame, text="Visualize Augmentations", 
                                    command=self.visualize_augmentations)
        visualize_aug_btn.pack(side=tk.LEFT, padx=5)

        compare_aug_btn = ttk.Button(random_buttons_frame, text="Compare Augmentations", 
                                    command=self.compare_augmentations)
        compare_aug_btn.pack(side=tk.LEFT, padx=5)

        # Pipeline controls
        pipeline_frame = ttk.LabelFrame(augmentation_tab, text="Augmentation Pipeline")
        pipeline_frame.pack(fill="x", expand=False, pady=5)

        # Pipeline severity selection
        pipeline_select_frame = ttk.Frame(pipeline_frame)
        pipeline_select_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(pipeline_select_frame, text="Pipeline Severity:").pack(side=tk.LEFT, padx=5)
        self.pipeline_severity_var = tk.StringVar(value="medium")
        pipeline_severities = ["light", "medium", "heavy"]
        pipeline_combo = ttk.Combobox(pipeline_select_frame, values=pipeline_severities, 
                                    textvariable=self.pipeline_severity_var, width=15)
        pipeline_combo.pack(side=tk.LEFT, padx=5)

        # Pipeline buttons
        pipeline_buttons_frame = ttk.Frame(pipeline_frame)
        pipeline_buttons_frame.pack(fill="x", padx=5, pady=5)

        preview_pipeline_btn = ttk.Button(pipeline_buttons_frame, text="Preview Pipeline", 
                                        command=self.preview_augmentation_pipeline)
        preview_pipeline_btn.pack(side=tk.LEFT, padx=5)

        # Dataset augmentation controls
        dataset_frame = ttk.LabelFrame(augmentation_tab, text="Dataset Augmentation")
        dataset_frame.pack(fill="x", expand=False, pady=5)

        # Augmentation factor
        factor_frame = ttk.Frame(dataset_frame)
        factor_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(factor_frame, text="Augmentation Factor:").pack(side=tk.LEFT, padx=5)
        self.aug_factor_var = tk.IntVar(value=5)
        factor_spinbox = ttk.Spinbox(factor_frame, from_=1, to=20, textvariable=self.aug_factor_var, width=10)
        factor_spinbox.pack(side=tk.LEFT, padx=5)

        # Dataset augmentation buttons
        dataset_buttons_frame = ttk.Frame(dataset_frame)
        dataset_buttons_frame.pack(fill="x", padx=5, pady=5)

        augment_dataset_btn = ttk.Button(dataset_buttons_frame, text="Augment Dataset", 
                                        command=self.augment_dataset)
        augment_dataset_btn.pack(side=tk.LEFT, padx=5)

        train_augmented_btn = ttk.Button(dataset_buttons_frame, text="Train with Augmentation", 
                                        command=self.train_with_augmentation)
        train_augmented_btn.pack(side=tk.LEFT, padx=5)

        # Tab for model testing (Week 13)
        testing_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(testing_tab, text="Model Testing")

        # Test dataset selection
        dataset_frame = ttk.LabelFrame(testing_tab, text="Test Dataset Selection")
        dataset_frame.pack(fill="x", expand=False, pady=5)

        # Dataset path
        dataset_path_frame = ttk.Frame(dataset_frame)
        dataset_path_frame.pack(fill="x", padx=5, pady=5)

        self.test_dataset_path_var = tk.StringVar()
        ttk.Label(dataset_path_frame, text="Test Dataset:").pack(side=tk.LEFT, padx=5)
        test_dataset_entry = ttk.Entry(dataset_path_frame, textvariable=self.test_dataset_path_var, width=50)
        test_dataset_entry.pack(side=tk.LEFT, padx=5)
        browse_test_dataset_btn = ttk.Button(dataset_path_frame, text="Browse", command=self.browse_test_dataset)
        browse_test_dataset_btn.pack(side=tk.LEFT, padx=5)

        # Output directory
        output_dir_frame = ttk.Frame(dataset_frame)
        output_dir_frame.pack(fill="x", padx=5, pady=5)

        self.test_output_dir_var = tk.StringVar()
        ttk.Label(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        test_output_entry = ttk.Entry(output_dir_frame, textvariable=self.test_output_dir_var, width=50)
        test_output_entry.pack(side=tk.LEFT, padx=5)
        browse_output_dir_btn = ttk.Button(output_dir_frame, text="Browse", command=self.browse_test_output_dir)
        browse_output_dir_btn.pack(side=tk.LEFT, padx=5)

        # Model selection for testing
        models_frame = ttk.LabelFrame(testing_tab, text="Models to Test")
        models_frame.pack(fill="x", expand=False, pady=5)

        self.test_models = {
            'cnn': tk.BooleanVar(value=True),
            'ml': tk.BooleanVar(value=True),
            'yolo': tk.BooleanVar(value=True)
        }

        models_grid = ttk.Frame(models_frame)
        models_grid.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(models_grid, text="CNN Models", variable=self.test_models['cnn']).grid(row=0, column=0, padx=10, pady=2)
        ttk.Checkbutton(models_grid, text="Traditional ML", variable=self.test_models['ml']).grid(row=0, column=1, padx=10, pady=2)
        ttk.Checkbutton(models_grid, text="YOLO", variable=self.test_models['yolo']).grid(row=0, column=2, padx=10, pady=2)

        # Testing options
        options_frame = ttk.LabelFrame(testing_tab, text="Testing Options")
        options_frame.pack(fill="x", expand=False, pady=5)

        self.test_with_error_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Include Error Analysis", 
                        variable=self.test_with_error_analysis_var).pack(anchor="w", padx=10, pady=2)

        self.efficiency_comparison_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Include Efficiency Comparison", 
                        variable=self.efficiency_comparison_var).pack(anchor="w", padx=10, pady=2)

        # Testing buttons
        testing_buttons_frame = ttk.Frame(testing_tab)
        testing_buttons_frame.pack(fill="x", expand=False, pady=5)

        run_test_btn = ttk.Button(testing_buttons_frame, text="Run Tests", command=self.run_model_tests)
        run_test_btn.pack(side=tk.LEFT, padx=5, pady=5)

        view_results_btn = ttk.Button(testing_buttons_frame, text="View Results", 
                                    command=self.view_test_results)
        view_results_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Tab for practical applications (Week 14)
        practical_tab = ttk.Frame(processing_tabs)
        processing_tabs.add(practical_tab, text="Practical Applications")

        # Real-time detection section
        realtime_frame = ttk.LabelFrame(practical_tab, text="Real-time Detection")
        realtime_frame.pack(fill="x", expand=False, pady=5)

        # Source selection
        source_frame = ttk.Frame(realtime_frame)
        source_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT, padx=5)
        self.rt_source_var = tk.StringVar(value="webcam")
        source_options = [("Webcam", "webcam"), ("Video File", "file"), ("RTSP Stream", "rtsp")]
        for text, value in source_options:
            ttk.Radiobutton(source_frame, text=text, variable=self.rt_source_var, value=value).pack(side=tk.LEFT, padx=5)

        # Source path/URL
        path_frame = ttk.Frame(realtime_frame)
        path_frame.pack(fill="x", padx=5, pady=5)

        self.rt_source_path_var = tk.StringVar()
        ttk.Label(path_frame, text="Path/URL:").pack(side=tk.LEFT, padx=5)
        source_entry = ttk.Entry(path_frame, textvariable=self.rt_source_path_var, width=50)
        source_entry.pack(side=tk.LEFT, padx=5)
        browse_source_btn = ttk.Button(path_frame, text="Browse", command=self.browse_rt_source)
        browse_source_btn.pack(side=tk.LEFT, padx=5)

        # Model selection for real-time
        model_frame = ttk.Frame(realtime_frame)
        model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.rt_model_var = tk.StringVar(value="yolo")
        rt_models = [("YOLO", "yolo"), ("CNN", "cnn"), ("Traditional ML", "traditional")]
        for text, value in rt_models:
            ttk.Radiobutton(model_frame, text=text, variable=self.rt_model_var, value=value).pack(side=tk.LEFT, padx=5)

        # Confidence threshold
        conf_frame = ttk.Frame(realtime_frame)
        conf_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.rt_confidence_var = tk.DoubleVar(value=0.5)
        conf_slider = ttk.Scale(conf_frame, from_=0.1, to=0.9, orient="horizontal", 
                            variable=self.rt_confidence_var, length=200)
        conf_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(conf_frame, textvariable=self.rt_confidence_var).pack(side=tk.LEFT, padx=5)

        # Real-time control buttons
        rt_buttons_frame = ttk.Frame(realtime_frame)
        rt_buttons_frame.pack(fill="x", padx=5, pady=5)

        self.start_rt_btn = ttk.Button(rt_buttons_frame, text="Start Detection", command=self.start_realtime_detection)
        self.start_rt_btn.pack(side=tk.LEFT, padx=5)

        self.stop_rt_btn = ttk.Button(rt_buttons_frame, text="Stop Detection", command=self.stop_realtime_detection, state=tk.DISABLED)
        self.stop_rt_btn.pack(side=tk.LEFT, padx=5)

        view_rt_stats_btn = ttk.Button(rt_buttons_frame, text="View Statistics", command=self.view_rt_statistics)
        view_rt_stats_btn.pack(side=tk.LEFT, padx=5)

        # Satellite image analysis section
        satellite_frame = ttk.LabelFrame(practical_tab, text="Satellite Image Analysis")
        satellite_frame.pack(fill="x", expand=False, pady=5)

        # Satellite image input
        sat_input_frame = ttk.Frame(satellite_frame)
        sat_input_frame.pack(fill="x", padx=5, pady=5)

        self.sat_image_path_var = tk.StringVar()
        ttk.Label(sat_input_frame, text="Image:").pack(side=tk.LEFT, padx=5)
        sat_entry = ttk.Entry(sat_input_frame, textvariable=self.sat_image_path_var, width=50)
        sat_entry.pack(side=tk.LEFT, padx=5)
        browse_sat_btn = ttk.Button(sat_input_frame, text="Browse", command=self.browse_satellite_image)
        browse_sat_btn.pack(side=tk.LEFT, padx=5)

        # Analysis parameters
        sat_params_frame = ttk.Frame(satellite_frame)
        sat_params_frame.pack(fill="x", padx=5, pady=5)

        # Tile size
        tile_frame = ttk.Frame(sat_params_frame)
        tile_frame.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(tile_frame, text="Tile Size:").pack(side=tk.LEFT, padx=5)
        self.tile_size_var = tk.IntVar(value=512)
        tile_sizes = [256, 512, 1024]
        tile_combo = ttk.Combobox(tile_frame, values=tile_sizes, textvariable=self.tile_size_var, width=10)
        tile_combo.pack(side=tk.LEFT, padx=5)

        # Overlap
        overlap_frame = ttk.Frame(sat_params_frame)
        overlap_frame.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(overlap_frame, text="Overlap:").pack(side=tk.LEFT, padx=5)
        self.overlap_var = tk.DoubleVar(value=0.2)
        overlap_slider = ttk.Scale(overlap_frame, from_=0.0, to=0.5, orient="horizontal", 
                                variable=self.overlap_var, length=150)
        overlap_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(overlap_frame, textvariable=self.overlap_var).pack(side=tk.LEFT, padx=5)

        # Confidence for satellite
        sat_conf_frame = ttk.Frame(sat_params_frame)
        sat_conf_frame.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(sat_conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.sat_confidence_var = tk.DoubleVar(value=0.5)
        sat_conf_slider = ttk.Scale(sat_conf_frame, from_=0.1, to=0.9, orient="horizontal", 
                                variable=self.sat_confidence_var, length=150)
        sat_conf_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(sat_conf_frame, textvariable=self.sat_confidence_var).pack(side=tk.LEFT, padx=5)

        # Output directory for satellite analysis
        output_frame = ttk.Frame(satellite_frame)
        output_frame.pack(fill="x", padx=5, pady=5)

        self.sat_output_dir_var = tk.StringVar()
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        output_entry = ttk.Entry(output_frame, textvariable=self.sat_output_dir_var, width=50)
        output_entry.pack(side=tk.LEFT, padx=5)
        browse_output_btn = ttk.Button(output_frame, text="Browse", command=self.browse_sat_output_dir)
        browse_output_btn.pack(side=tk.LEFT, padx=5)

        # Analysis buttons
        sat_buttons_frame = ttk.Frame(satellite_frame)
        sat_buttons_frame.pack(fill="x", padx=5, pady=5)

        analyze_btn = ttk.Button(sat_buttons_frame, text="Analyze Image", command=self.analyze_satellite_image)
        analyze_btn.pack(side=tk.LEFT, padx=5)

        view_sat_results_btn = ttk.Button(sat_buttons_frame, text="View Results", command=self.view_satellite_results)
        view_sat_results_btn.pack(side=tk.LEFT, padx=5)
        
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
        
        # Status bar for classification results
        self.status_frame = ttk.Frame(right_panel)
        self.status_frame.pack(fill=tk.X, expand=False, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
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

    # Classification-related methods
    def train_classifier(self):
        """Train selected classifier type"""
        classifier_type = self.classifier_type_var.get()
        
        # Get dataset directory from user
        dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if not dataset_dir:
            return
        
        try:
            if classifier_type == "Traditional":
                # Traditional ML classifier training
                model_type = self.model_var.get()
                feature_method = self.feature_extractor_var.get()
                
                self.status_var.set(f"Training {model_type} model with {feature_method} features...")
                self.root.update()
                
                accuracy, report = self.image_loader.train_classifier(
                    dataset_dir, 
                    model_type=model_type, 
                    feature_method=feature_method
                )
                
                # Show results
                result_dialog = ClassifierResultDialog(self.root, 
                                                   model_type=model_type,
                                                   feature_method=feature_method,
                                                   accuracy=accuracy,
                                                   report=report)
                
                self.status_var.set(f"Model trained: {model_type} (Accuracy: {accuracy:.2f})")
                
            elif classifier_type == "CNN":
                # CNN training
                epochs = self.epochs_var.get()
                batch_size = self.batch_size_var.get()
                
                self.status_var.set(f"Training CNN model (Epochs: {epochs}, Batch Size: {batch_size})...")
                self.root.update()
                
                history = self.image_loader.train_cnn_classifier(
                    dataset_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2
                )
                
                # Get final training metrics
                final_accuracy = history.history['accuracy'][-1]
                final_val_accuracy = history.history['val_accuracy'][-1]
                
                self.status_var.set(f"CNN trained - Accuracy: {final_accuracy:.3f}, Val Accuracy: {final_val_accuracy:.3f}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")
    
    def classify_image(self):
        """Classify using selected classifier type"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        classifier_type = self.classifier_type_var.get()
        
        try:
            if classifier_type == "Traditional":
                if self.image_loader.classifier.model is None:
                    messagebox.showerror("Error", "No traditional model available")
                    return
                    
                predicted_class, confidence = self.image_loader.classify_image()
                
            elif classifier_type == "CNN":
                if self.image_loader.cnn_classifier.model is None:
                    messagebox.showerror("Error", "No CNN model available")
                    return
                    
                predicted_class, confidence = self.image_loader.classify_image_cnn()
            
            # Display result on image
            result_image = self.image_loader.draw_classification_result(predicted_class, confidence)
            self.current_image = result_image
            self.display_image(result_image)
            
            # Update status
            conf_str = f" (Confidence: {confidence:.2f})" if confidence is not None else ""
            self.status_var.set(f"Classification result: {predicted_class}{conf_str}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_var.set("Classification failed")
    
    def show_training_history(self):
        """Show CNN training history plots"""
        classifier_type = self.classifier_type_var.get()
        
        if classifier_type == "CNN":
            try:
                self.image_loader.plot_cnn_training_history()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show training history: {str(e)}")
        else:
            messagebox.showinfo("Info", "Training history is only available for CNN models")
    
    def show_model_summary(self):
        """Show CNN model architecture summary"""
        classifier_type = self.classifier_type_var.get()
        
        if classifier_type == "CNN":
            try:
                summary = self.image_loader.get_cnn_summary()
                # Create a dialog to display the summary
                summary_dialog = ModelSummaryDialog(self.root, summary)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show model summary: {str(e)}")
        else:
            messagebox.showinfo("Info", "Model summary is only available for CNN models")
    
    def load_classifier_model(self):
        """Load model based on selected classifier type"""
        classifier_type = self.classifier_type_var.get()
        
        if classifier_type == "Traditional":
            # Load traditional ML model (.pkl)
            model_path = filedialog.askopenfilename(
                title="Load Model",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
        else:  # CNN
            # Load CNN model files
            model_path = filedialog.askopenfilename(
                title="Load CNN Model",
                filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
            )
            if model_path and model_path.endswith('.h5'):
                # Remove .h5 extension for loading
                model_path = model_path[:-3]
        
        if not model_path:
            return
        
        try:
            if classifier_type == "Traditional":
                self.image_loader.load_classifier_model(model_path)
                
                # Update UI
                if hasattr(self.image_loader.classifier, 'model_type'):
                    self.model_var.set(self.image_loader.classifier.model_type)
                    
                if hasattr(self.image_loader.classifier, 'feature_extractor'):
                    self.feature_extractor_var.set(self.image_loader.classifier.feature_extractor)
                    
            else:  # CNN
                self.image_loader.load_cnn_model(model_path)
            
            # Update status
            self.status_var.set(f"Model loaded from: {os.path.basename(model_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed")
    
    def save_classifier_model(self):
        """Save model based on selected classifier type"""
        classifier_type = self.classifier_type_var.get()
        
        if classifier_type == "Traditional":
            if self.image_loader.classifier.model is None:
                messagebox.showerror("Error", "No traditional model to save")
                return
                
            model_path = filedialog.asksaveasfilename(
                title="Save Model",
                defaultextension=".pkl",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
        else:  # CNN
            if self.image_loader.cnn_classifier.model is None:
                messagebox.showerror("Error", "No CNN model to save")
                return
                
            model_path = filedialog.asksaveasfilename(
                title="Save CNN Model",
                defaultextension=".h5",
                filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
            )
            if model_path and model_path.endswith('.h5'):
                # Remove .h5 extension for saving
                model_path = model_path[:-3]
        
        if not model_path:
            return
        
        try:
            if classifier_type == "Traditional":
                self.image_loader.save_classifier_model(model_path)
            else:  # CNN
                self.image_loader.save_cnn_model(model_path)
            
            # Update status
            self.status_var.set(f"Model saved to: {os.path.basename(model_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            self.status_var.set("Model saving failed")

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

    def process_image(self, process_type):
        """Process the current image based on the selected method."""
        if self.current_image is None:
            messagebox.showerror("Error", "No image loaded")
            return
        
        if process_type == 'pretrained':
            # This is handled by the specific pre-trained model methods
            pass
            
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


class ClassifierResultDialog(tk.Toplevel):
    """Dialog to display classifier training results."""
    def __init__(self, parent, model_type, feature_method, accuracy, report):
        super().__init__(parent)
        self.title("Classifier Training Results")
        self.geometry("500x400")
        self.resizable(True, True)
        
        # Main frame
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill="both", expand=True)
        
        # Model info
        info_frame = ttk.LabelFrame(frame, text="Model Information")
        info_frame.pack(fill="x", expand=False, pady=5)
        
        ttk.Label(info_frame, text=f"Model Type: {model_type}").pack(anchor="w", padx=5, pady=2)
        ttk.Label(info_frame, text=f"Feature Extractor: {feature_method}").pack(anchor="w", padx=5, pady=2)
        ttk.Label(info_frame, text=f"Accuracy: {accuracy:.4f}").pack(anchor="w", padx=5, pady=2)
        
        # Classification report
        report_frame = ttk.LabelFrame(frame, text="Classification Report")
        report_frame.pack(fill="both", expand=True, pady=5)
        
        # Text widget for the report
        report_text = tk.Text(report_frame, wrap="none", height=15)
        report_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
        y_scrollbar.pack(side="right", fill="y")
        
        x_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=report_text.xview)
        x_scrollbar.pack(fill="x")
        
        report_text.config(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Insert the report
        report_text.insert("1.0", report)
        report_text.config(state="disabled")  # Make read-only
        
        # Close button
        ttk.Button(frame, text="Close", command=self.destroy).pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() // 2) - (width // 2) + parent.winfo_x()
        y = (parent.winfo_height() // 2) - (height // 2) + parent.winfo_y()
        self.geometry(f"{width}x{height}+{x}+{y}")


class ModelSummaryDialog(tk.Toplevel):
    """Dialog to display CNN model summary."""
    def __init__(self, parent, summary_text):
        super().__init__(parent)
        self.title("CNN Model Summary")
        self.geometry("600x400")
        self.resizable(True, True)
        
        # Main frame
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill="both", expand=True)
        
        # Text widget for the summary
        summary_display = tk.Text(frame, wrap="none", width=60, height=20)
        summary_display.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=summary_display.yview)
        y_scrollbar.pack(side="right", fill="y")
        
        x_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=summary_display.xview)
        x_scrollbar.pack(side="bottom", fill="x")
        
        summary_display.config(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Insert the summary
        summary_display.insert("1.0", summary_text)
        summary_display.config(state="disabled")  # Make read-only
        
        # Close button
        ttk.Button(frame, text="Close", command=self.destroy).pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() // 2) - (width // 2) + parent.winfo_x()
        y = (parent.winfo_height() // 2) - (height // 2) + parent.winfo_y()
        self.geometry(f"{width}x{height}+{x}+{y}")


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