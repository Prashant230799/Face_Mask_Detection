import cv2
import tensorflow as tf
import numpy as np

# ==========================================
# Load Trained Model
# ==========================================

model = tf.keras.models.load_model("models/facemask_detector.keras")
print("✅ Model Loaded Successfully!")

# ==========================================
# Load Face Detector
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("✅ Face Detector Loaded Successfully!")

# ==========================================
# Open Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot Open Webcam")
    exit()

print("✅ Webcam Started!")

# ==========================================
# Webcam Loop
# ==========================================

while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Loop through detected faces
    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize
        face = cv2.resize(face, (224, 224))

        # Normalize
        face = face.astype(np.float32) / 255.0

        # Convert to batch
        face = np.expand_dims(face, axis=0)

        # ===============================
        # Prediction
        # ===============================

        prediction = model.predict(face, verbose=0)

        probability = prediction[0][0]

        # ==================================================
        # IMPORTANT:
        # Your training labels were:
        # with_mask = 1
        # without_mask = 0
        # ==================================================

        if probability >= 0.5:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Display label
        cv2.putText(
            frame,
            f"{label} ({probability:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Show webcam
    cv2.imshow("Face Mask Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# Release Resources
# ==========================================

cap.release()
cv2.destroyAllWindows()