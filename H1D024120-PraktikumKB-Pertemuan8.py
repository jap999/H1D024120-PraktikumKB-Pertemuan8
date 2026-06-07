import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Menentukan direktori data secara manual sesuai folder lokal laptop kamu
base_dir = 'rockpaperscissors/rps-cv-images'

# Menggunakan ImageDataGenerator untuk augmentasi data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='wrap',
    validation_split=0.4 # Membagi data validasi 40%
)

# Generator data latihan
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Generator data validasi
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Membangun arsitektur model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model menggunakan optimizer Adam
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Melatih model dengan fit
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Mengevaluasi performa model
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss} , Validation accuracy: {val_acc}')

# Memprediksi hasil model
predictions = model.predict(validation_generator)
print(predictions)


# ==========================================
# PART 2: TAMBAHAN FITUR (GRAFIK MONITORS)
# ==========================================

import matplotlib.pyplot as plt

# Menampilkan grafik akurasi dan loss agar laporan responsi makin lengkap
plt.figure(figsize=(12, 4))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Grafik Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
