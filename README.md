# Enhancing-an-HCI-Application-Based-on-a-Use-Case
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Detection parameters
conf_threshold = 0.5  # Minimum confidence to consider a detection
nms_threshold = 0.4   # Threshold for Non-Maximum Suppression

# File paths for YOLO (make sure these files are in the same directory as the script)
yolo_config_path = r"C:\Users\ardar\OneDrive\Escritorio\YOLO use case\yolov3.cfg.txt"         # Network architecture file (Darknet format)
yolo_weights_path = r"C:\Users\ardar\OneDrive\Escritorio\YOLO use case\yolov3.weights"        # Pre-trained weights file
classes_file = r"C:\Users\ardar\OneDrive\Escritorio\YOLO use case\coco.names.txt"             # File with class names from the COCO dataset

# Load class names
try:
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    messagebox.showerror("Error", f"File not found: {classes_file}.")
    exit()

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("YOLO model loaded successfully.")

# Get the output layer names of the network
layer_names = net.getLayerNames()
# In recent OpenCV versions, getUnconnectedOutLayers() returns a 1D array
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_image(image_path):
    """Loads an image, performs object detection with YOLO, and returns the processed image."""
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image.")
        return None

    (height, width) = image.shape[:2]
    # Create a blob from the original image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf_text = f"{confidences[i]*100:.2f}%"
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label}: {conf_text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def load_and_process_image():
    """Triggered when the 'Load Image' button is clicked.
    Opens a file dialog to select an image, processes it, and displays it in the interface."""
    file_path = filedialog.askopenfilename(title="Select an image",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        processed = process_image(file_path)
        if processed is not None:
            # Convert image from BGR to RGB and then to a PIL object to display in Tkinter
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(processed_rgb)
            # Resize image for display (optional)
            pil_image = pil_image.resize((800, 600))
            imgtk = ImageTk.PhotoImage(image=pil_image)
            label_img.config(image=imgtk)
            label_img.image = imgtk

# Create the main application window
root = tk.Tk()
root.title("Object Detection with YOLO - HCI Application")

# Button to load and process the image
btn_load = tk.Button(root, text="Load Image", command=load_and_process_image)
btn_load.pack(pady=10)

# Label to display the processed image
label_img = tk.Label(root)
label_img.pack(padx=10, pady=10)

# Start the application
root.mainloop()
