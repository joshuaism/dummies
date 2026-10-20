import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model created using laranel_detector_program on google colab
# based on data in datasets_and_media/laranel_detector directory
# annotated using https://www.cvat.ai/
model = YOLO("datasets_and_media/best.pt")

# Open video capture
cap = cv2.VideoCapture(0)

key = ord("r")

while key != ord("s") and key != ord("S"):
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run LOLOv8 inference on the frame
        results = model(frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        key = cv2.waitKey(5)
cv2.destroyAllWindows()
