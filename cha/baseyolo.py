from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Load an image for object detection
image_path = "xxx.jpg"
results = model(image_path)  # Perform object detection

# Display the detection results
for result in results:
    img = result.plot()  # Draw bounding boxes and labels on the image
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
