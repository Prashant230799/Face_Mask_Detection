import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model

# =====================================================
# 1. Read CSV
# =====================================================

df = pd.read_csv("dataset/labels.csv")

# =====================================================
# 2. Load and Preprocess Images
# =====================================================

images = []
labels = []

for _, row in df.iterrows():

    # Image Path
    image_path = os.path.join("dataset", "images", row["filename"])

    # Read Image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping: {row['filename']}")
        continue

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize Image
    image = cv2.resize(image, (224, 224))

    # Normalize Image
    image = image / 255.0

    # Store Image
    images.append(image)

    # Convert One-Hot Label → Binary Label
    if row["with_mask"] == 1:
        labels.append(1)      # Mask
    else:
        labels.append(0)      # No Mask

# =====================================================
# 3. Convert to NumPy Arrays
# =====================================================

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

# =====================================================
# 4. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# 5. Load MobileNetV2
# =====================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print("\n✅ MobileNetV2 Loaded Successfully!")

# =====================================================
# 6. Freeze MobileNetV2
# =====================================================

base_model.trainable = False

print("✅ Base Model Frozen Successfully!")

# =====================================================
# 7. Build Custom Model
# =====================================================

inputs = Input(shape=(224, 224, 3))

# Feature Extraction
x = base_model(inputs, training=False)

# Reduce Feature Maps
x = GlobalAveragePooling2D()(x)

# Reduce Overfitting
x = Dropout(0.2)(x)

# Output Layer
outputs = Dense(1, activation="sigmoid")(x)

# Final Model
model = Model(inputs=inputs, outputs=outputs)

# =====================================================
# 8. Model Summary
# =====================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# =====================================================
# 10. Compile Model
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n✅ Model Compiled Successfully!")

# =====================================================
# 11. Train Model
# =====================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

print("\n✅ Model Training Completed!")

# =====================================================
# 12. Evaluate Model
# =====================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# =====================================================
# 13. Save Model
# =====================================================

import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/facemask_detector.keras")

print("\n✅ Model Saved Successfully!")
print("Location: models/facemask_detector.keras")

# =====================================================
# 9. Dataset Information
# =====================================================



print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Original Images Shape : {X.shape}")
print(f"Original Labels Shape : {y.shape}")

print(f"\nTraining Images Shape : {X_train.shape}")
print(f"Training Labels Shape : {y_train.shape}")

print(f"\nTesting Images Shape : {X_test.shape}")
print(f"Testing Labels Shape : {y_test.shape}")

print("\nFirst 10 Labels:")
print(y[:10])