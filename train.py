import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.dataset import check_dataset

# Check if the dataset is present
dataset_path = 'dataset'
check_dataset('dataset')
train_directory = 'dataset/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
test_directory = 'dataset/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

# Data preprocessing and visualization
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Visualize some training images
plt.figure(figsize=(10, 10))
for images, labels in train_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        # Get the class label from the one-hot encoded labels
        predicted_class = np.argmax(labels[i], axis=-1)
        class_label = list(train_generator.class_indices.keys())[predicted_class]
        plt.title(f"Class: {class_label}")
        plt.axis("off")
    break
plt.show()

# Build the CNN model
model = Sequential([
    layers.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

print("Class Indices:", train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())
print("Class Labels:", class_labels)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
if not os.path.exists('models'): os.mkdir('models')
if not os.path.exists('models/checkpoints'): os.mkdir('models/checkpoints')
# Train the model with visualization
print(model.summary())
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='auto'),
        ModelCheckpoint(
            "models/checkpoints/skin_cancer_detection2_checkpoint.keras",
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            mode='auto',
            save_weights_only=False
        )
    ]
)

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save('models/trained_cancer_cnn_2.keras')