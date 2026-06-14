import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load OpenCV's built-in face detector and your trained mask model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("mask_detector.h5")

print("[INFO] Starting video stream... Press 'q' to quit.")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale for the face bounding-box detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face exactly how we processed the training dataset
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = preprocess_input(face_roi)  # Scales between -1 and 1

        # Make the mask prediction
        (mask, withoutMask) = model.predict(face_roi, verbose=0)[0]

        # Determine class label and box color (BGR format)
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green for Mask, Red for No Mask

        # Format label text with the prediction confidence percentage
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        # Draw the bounding box and text on the frame
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show the interactive window
    cv2.imshow("Real-Time Face Mask Detector", frame)

    # Hit 'q' on your keyboard to close out the stream safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up windows and camera streams safely
video_capture.release()
cv2.destroyAllWindows()