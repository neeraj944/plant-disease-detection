import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



# PARAMETERS

DATASET_DIR = "dataset/PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5              # Initial training epochs
FINE_TUNE_EPOCHS = 5    # Fine-tuning epochs



# DATA LOADING & AUGMENTATION

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes
print("Number of classes:", NUM_CLASSES)



# MODEL (TRANSFER LEARNING)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all base model layers (Stage 1)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# STAGE 1: INITIAL TRAINING

print("\n Starting initial training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)



# STAGE 2: FINE-TUNING

print("\n Starting fine-tuning...\n")

# Unfreeze last 30 layers of MobileNetV2
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS
)



# SAVE FINAL MODEL

model.save("plant_disease_model_finetuned.h5")
print(" Fine-tuned model saved as plant_disease_model_finetuned.h5")



# PLOT ACCURACY & LOSS

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history_fine.history['accuracy'], label='Fine-tune Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Fine-tune Val Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history_fine.history['loss'], label='Fine-tune Train Loss')
plt.plot(history_fine.history['val_loss'], label='Fine-tune Val Loss')
plt.title('Loss')
plt.legend()

plt.show()
