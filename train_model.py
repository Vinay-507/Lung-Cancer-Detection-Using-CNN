import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# ==============================
# PATHS (Do not change)
# ==============================
TRAIN_DIR = "Data/train"
VALID_DIR = "Data/valid"
TEST_DIR  = "Data/test"
MODEL_DIR = "model"

# create model folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# PARAMETERS
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4   # Adenocarcinoma, Large cell, Squamous cell, Normal

# ==============================
# DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\nCLASS INDICES:", train_data.class_indices)

# ==============================
# MODEL ARCHITECTURE
# ==============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# SAVE BEST MODEL
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_cnn.h5")

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ==============================
# TRAINING
# ==============================
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("\nTraining Completed Successfully!")
print(f"Best model saved at: {MODEL_PATH}")

# ==============================
# PLOT ACCURACY & LOSS
# ==============================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.legend()

GRAPH_PATH = os.path.join(MODEL_DIR, "training_graphs.png")
plt.savefig(GRAPH_PATH)

print(f"Training Graph Saved at: {GRAPH_PATH}")
plt.show()
