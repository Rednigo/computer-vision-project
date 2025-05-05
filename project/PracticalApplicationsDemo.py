import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
from datetime import datetime
from RealTimeDetection import RealTimeDetection
from SatelliteImageAnalysis import SatelliteImageAnalysis

def demonstrate_realtime_detection():
    """Demonstrate real-time aerial object detection"""
    print("Real-time Detection Demonstration")
    print("=" * 40)
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user for source
    source_options = ["Webcam", "Video File", "Exit"]
    source_choice = None
    
    print("\nSelect detection source:")
    for i, option in enumerate(source_options):
        print(f"{i+1}. {option}")
    
    try:
        choice = int(input("Enter choice (1-3): "))
        if 1 <= choice <= len(source_options):
            source_choice = source_options[choice-1]
    except:
        print("Invalid choice")
        return
    
    if source_choice == "Exit":
        return
    
    # Get source
    if source_choice == "Webcam":
        source = 0
    else:  # Video File
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if not video_path:
            print("No video file selected")
            return
        source = video_path
    
    try:
        # Initialize detection system
        print("\nInitializing real-time detection system...")
        detector = RealTimeDetection(primary_model='yolo')
        
        # Configure
        detector.configure(
            confidence_threshold=0.5,
            save_detections=True,
            save_directory="realtime_detections_demo"
        )
        
        print("Starting real-time detection...")
        print("Press 'q' to quit")
        
        # Start detection
        detector.start_detection(source)
        
        # Generate report
        report = detector.generate_report("realtime_detection_report.json")
        
        print("\nReal-time Detection Summary:")
        print(f"Total frames processed: {report['summary']['total_frames']}")
        print(f"Total detections: {report['summary']['total_detections']}")
        print(f"Average FPS: {report['summary']['average_fps']:.1f}")
        print("\nClass Distribution:")
        for class_name, count in report['class_distribution'].items():
            print(f"  {class_name}: {count}")
        
        # Visualize statistics
        detector.visualize_statistics("realtime_stats_demo.png")
        print("\nStatistics visualization saved to realtime_stats_demo.png")
        
    except Exception as e:
        print(f"Error in real-time detection: {e}")

def demonstrate_satellite_analysis():
    """Demonstrate satellite image analysis"""
    print("\nSatellite Image Analysis Demonstration")
    print("=" * 40)
    
    # Create Tkinter root and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user for satellite image
    image_path = filedialog.askopenfilename(
        title="Select Satellite/Aerial Image",
        filetypes=[
            ("GeoTIFF files", "*.tif *.tiff"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("No image selected")
        return
    
    # Create output directory
    output_dir = "satellite_analysis_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\nAnalyzing satellite image...")
        print(f"Input image: {os.path.basename(image_path)}")
        print(f"Output directory: {output_dir}")
        
        # Initialize analysis system
        analyzer = SatelliteImageAnalysis(tile_size=512, overlap=0.2)
        
        # Perform analysis
        results = analyzer.analyze_satellite_image(
            image_path,
            output_dir,
            confidence_threshold=0.5
        )
        
        # Create interactive HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analyzer.create_interactive_html_report(image_path, output_dir, timestamp)
        
        print("\nSatellite Analysis Results:")
        print(f"Detections found: {results['detection_count']}")
        print("\nClass Distribution:")
        for class_name, count in results['class_distribution'].items():
            print(f"  {class_name}: {count}")
        
        print(f"\nResults saved to: {output_dir}")
        
        # Ask if user wants to view report
        if messagebox.askyesno("View Report", "Open HTML report in browser?"):
            import webbrowser
            report_path = os.path.join(output_dir, f"report_{timestamp}.html")
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
        
    except Exception as e:
        print(f"Error in satellite analysis: {e}")

def demonstrate_both_applications():
    """Demonstrate both real-time and satellite analysis"""
    print("Comprehensive Practical Applications Demonstration")
    print("=" * 50)
    
    while True:
        print("\nSelect demonstration:")
        print("1. Real-time Detection")
        print("2. Satellite Image Analysis")
        print("3. Exit")
        
        try:
            choice = int(input("Enter choice (1-3): "))
            
            if choice == 1:
                demonstrate_realtime_detection()
            elif choice == 2:
                demonstrate_satellite_analysis()
            elif choice == 3:
                break
            else:
                print("Invalid choice")
                
        except ValueError:
            print("Invalid input")
        except Exception as e:
            print(f"Error: {e}")

def create_comparison_visualization():
    """Create visualization comparing traditional and real-time approaches"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Traditional single-image analysis
    methods = ['CNN', 'Traditional ML', 'YOLO']
    accuracy = [0.92, 0.85, 0.88]
    processing_time = [0.3, 0.1, 0.05]
    
    ax1.bar(methods, accuracy, color=['#2E86C1', '#28B463', '#F39C12'])
    ax1.set_title('Traditional Single Image Analysis')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    
    for i, (acc, time) in enumerate(zip(accuracy, processing_time)):
        ax1.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
        ax1.text(i, 0.05, f'{time:.2f}s', ha='center', color='white', fontweight='bold')
    
    # Real-time performance
    models = ['YOLO', 'Lightweight CNN', 'Traditional ML']
    fps = [30, 15, 45]
    latency = [0.033, 0.067, 0.022]
    
    ax2.bar(models, fps, color=['#F39C12', '#3498DB', '#28B463'])
    ax2.set_title('Real-time Performance')
    ax2.set_ylabel('FPS')
    
    for i, (f, lat) in enumerate(zip(fps, latency)):
        ax2.text(i, f + 1, f'{f} FPS', ha='center')
        ax2.text(i, f/2, f'{lat*1000:.0f}ms', ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('practical_applications_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create system architecture diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create blocks
    blocks = {
        'Input': (1, 4, '#3498DB'),
        'Preprocessing': (3, 4, '#2ECC71'),
        'Model Selection': (5, 4, '#F39C12'),
        'Detection/Classification': (7, 4, '#E74C3C'),
        'Output': (9, 4, '#9B59B6')
    }
    
    for text, (x, y, color) in blocks.items():
        rect = plt.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add arrows
    for i in range(1, 9, 2):
        ax.arrow(i+0.8, 4, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Add model options
    models = ['YOLO', 'CNN', 'ML']
    for i, model in enumerate(models):
        y_pos = 2 - i*0.4
        rect = plt.Rectangle((4.2, y_pos-0.15), 1.6, 0.3, 
                           facecolor='lightgray', edgecolor='black')
        ax.add_patch(rect)
        ax.text(5, y_pos, model, ha='center', va='center', fontsize=10)
        ax.arrow(5, y_pos+0.15, 0, 1.85-y_pos, head_width=0.1, head_length=0.1, 
                fc='gray', ec='gray', linestyle='dashed')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Aerial Object Recognition System Architecture', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison visualizations created successfully")

def main():
    """Main demonstration function"""
    print("Practical Applications of Aerial Object Recognition")
    print("================================================")
    
    # Create comparison visualizations
    create_comparison_visualization()
    
    # Run main demonstration
    demonstrate_both_applications()
    
    print("\nDemonstration completed!")
    print("Output files generated:")
    print("  - realtime_stats_demo.png")
    print("  - practical_applications_comparison.png")
    print("  - system_architecture.png")
    print("  - realtime_detections_demo/ (folder)")
    print("  - satellite_analysis_demo/ (folder)")

if __name__ == "__main__":
    main()