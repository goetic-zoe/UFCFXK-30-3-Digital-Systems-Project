import os

from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.ops.gen_data_flow_ops import queue_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Augmentor
import numpy as np
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Dynamic memory growth enabled")
    except RuntimeError as e:
        print(e)

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input

from utils.dataset import check_dataset

check_dataset('dataset')

print("--- TensorFlow GPU Verification ---")
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

if gpus:
    print("Success! TensorFlow can access your GPU via PyCharm.")
else:
    print("Failed. PyCharm is still missing the required CUDA library paths.")

# Define paths
train_dir = 'dataset/Skin cancer ISIC The International Skin Imaging Collaboration/Train/'
checkpoint_path = 'models/checkpoints/skin_CNN2_CPU.keras'
model_save_path = 'models/skin_CNN2_CPU.keras'

# img_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir)
#
# for i in img_dataset.class_names:
#     p = Augmentor.Pipeline(train_dir + i, output_directory='../' + i)
#     p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#     p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
#     p.flip_left_right(probability=0.5)
#     p.shear(probability=0.5, max_shear_left=20, max_shear_right=20)
#     p.random_brightness(probability=0.3, min_factor=0.8, max_factor=1.2)
#     p.random_color(probability=0.3, min_factor=0.8, max_factor=1.2)
#     p.sample(1000)

BATCH_SIZE = 32
EPOCHS = 50

# 1. ENHANCED AUGMENTATION FOR SKIN LESIONS
# Skin marks require rotation flexibility and zoom variance
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       # Increased from 5 to 30
    width_shift_range=0.1,    # Handles centering differences
    height_shift_range=0.1,
    zoom_range=0.15,          # Handles macro vs close-up lenses
    horizontal_flip=True,
    vertical_flip=True,       # Crucial for non-directional skin images
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(180, 180),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(180, 180),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. ARCHITECTURE (Stable 4-Block Setup)
model = Sequential([
    Input(shape=(180, 180, 3)),

    # Block 1 - Base Edges (Using string-based leaky_relu)
    Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 2 - Shapes
    Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 3 - Borders
    Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 4 - Micro-textures
    Conv2D(256, (3, 3), activation='leaky_relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 5 - Fine Pigment Detail (NO MAX POOLING)
    Conv2D(512, (3, 3), activation='leaky_relu', padding='same'),
    BatchNormalization(),

    # Compress the remaining spatial features smoothly
    GlobalAveragePooling2D(),

    # Classifier
    Dense(256, activation='leaky_relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# 3. STABILIZED OPTIMIZER
# We explicitly set Adam's learning rate and add clipnorm to stop loss explosions
opt = tf.keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1.0)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# 4. TRAINING DYNAMICS (Essential for smoothing the waves)
if not os.path.exists('models'): os.mkdir('models')
if not os.path.exists('models/checkpoints'): os.mkdir('models/checkpoints')

callbacks = [
    ModelCheckpoint(
        "models/checkpoints/skin_CNN2.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # INCREASED PATIENCE: Gives the model 4 epochs to settle instead of 2 before
    # cutting the learning rate. This prevents it from dropping down to 1e-6 too early.
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,       # Mild drop (30%) instead of a harsh 20% drop
        patience=4,       # Look for stagnation across 4 epochs
        min_lr=1e-6,
        verbose=1
    ),
    # Increased patience to 8 so training isn't cut short prematurely
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

# Run training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks,
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

# Save the final model
model.save(model_save_path)