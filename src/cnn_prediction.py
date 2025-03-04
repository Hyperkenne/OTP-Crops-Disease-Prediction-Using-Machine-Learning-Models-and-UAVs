import os
import shutil
import zipfile
import numpy as np
import cv2  # OpenCV for image preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define paths
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'dataset')
processed_dir = os.path.join(base_dir, 'processed_dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
model_dir = os.path.join(base_dir, 'models')
output_dir = os.path.join(base_dir, 'outputs')

# Ensure required directories exist
for directory in [dataset_dir, processed_dir, train_dir, val_dir, model_dir, output_dir]:
    os.makedirs(directory, exist_ok=True)

# Step 1: Extract Dataset
zip_path = os.path.join(base_dir, 'dataset.zip')  # Ensure correct path
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

# Step 2: Preprocess Images (Resize & Normalize)
img_size = (224, 224)  # Resize images for model compatibility
class_names = os.listdir(dataset_dir)

for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)
    processed_class_path = os.path.join(processed_dir, class_name)
    os.makedirs(processed_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, img_size)  # Resize
            img = img / 255.0  # Normalize

            save_path = os.path.join(processed_class_path, img_name)
            cv2.imwrite(save_path, img * 255)  # Save normalized image

# Step 3: Split Dataset into Train and Validation
for class_name in class_names:
    class_path = os.path.join(processed_dir, class_name)
    images = os.listdir(class_path)

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img_name in train_images:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(train_class_dir, img_name))

    for img_name in val_images:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(val_class_dir, img_name))

# Step 4: Load Data for Model Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=32, class_mode='categorical')

# Step 5: Build Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Unfreeze layers for fine-tuning
for layer in base_model.layers[:-20]:  # Freeze early layers, fine-tune deeper layers
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train Model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_generator, validation_data=val_generator, epochs=80, steps_per_epoch=15, validation_steps=15, callbacks=[reduce_lr, early_stop])

# Save Model
model.save(os.path.join(model_dir, 'plant_disease_model.h5'))

# Load trained model
model = load_model(os.path.join(model_dir, 'plant_disease_model.h5'))

# Prepare validation data generator
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=32, class_mode='categorical', shuffle=False)

# Get true labels and class names
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Predict using the trained model
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Save confusion matrix to Excel
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
conf_matrix_df.to_excel(os.path.join(output_dir, 'confusion_matrix.xlsx'))

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
