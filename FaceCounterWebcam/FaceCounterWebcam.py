import os
import sys
import subprocess

# -------------------------------
# Auto-install missing modules
# -------------------------------
try:
    import cv2
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

try:
    import numpy as np
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

# -------------------------------
# Paths to DNN model files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Check if model files exist
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("Error: Model files not found. Please put 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' in the project folder.")
    sys.exit()

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam")
    sys.exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # Prepare input for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw rectangle around human face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_count += 1

    # Display face count
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video
    cv2.imshow("Human Face Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
