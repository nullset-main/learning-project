import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from features import IMG_SIZE

# Model Loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "hand_sign_cnn.h5")
model = load_model(model_path)
print("Model loaded!")

# Labelling
labels = {0: "scissors", 1: "paper", 2: "rock"}

camera_index = None
for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ret, _ = cap.read()
    if ret:
        camera_index = i
        print(f"Using camera index {i}")
        break
    cap.release()

if camera_index is None:
    raise RuntimeError("No camera found. Check macOS permissions!")

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess
    roi_resized = cv2.resize(roi, IMG_SIZE)
    roi_normalized = roi_resized / 255.0
    img_array = np.reshape(roi_normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 3))

    # Predict
    pred = model.predict(img_array, verbose=0)
    class_index = np.argmax(pred)
    confidence = np.max(pred)
    label = labels.get(class_index, "Unknown")

    # Display
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
