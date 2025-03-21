import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from AerialImageLoader import AerialImageLoader

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.image_loader = AerialImageLoader()
        self.image_paths = []
        self.current_image = None

        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.load_button = ttk.Button(self.frame, text="Load Images", command=self.load_images)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.image_listbox = tk.Listbox(self.frame, height=10)
        self.image_listbox.grid(row=1, column=0, padx=5, pady=5)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.action_combobox = ttk.Combobox(self.frame, values=["Denoise", "Sharpen", "Threshold Segmentation", "Otsu Segmentation", "Watershed Segmentation", "GrabCut Segmentation", "Detect Contours", "Detect Features (SIFT)", "Detect Features (ORB)", "Detect Features (HOG)"])
        self.action_combobox.grid(row=2, column=0, padx=5, pady=5)
        self.action_combobox.set("Select Action")

        self.process_button = ttk.Button(self.frame, text="Process", command=self.process_image)
        self.process_button.grid(row=3, column=0, padx=5, pady=5)

        self.image_label = ttk.Label(self.frame)
        self.image_label.grid(row=4, column=0, padx=5, pady=5)

    def load_images(self):
        self.image_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, path)

    def on_image_select(self, event):
        selected_index = self.image_listbox.curselection()
        if selected_index:
            image_path = self.image_listbox.get(selected_index)
            self.current_image = self.image_loader.load_image(image_path)
            self.display_image(self.current_image)

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

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

        self.display_image(result_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()