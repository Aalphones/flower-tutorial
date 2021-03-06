import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import argparse

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Parse arguments
parser = argparse.ArgumentParser(description="Training CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-it", "--iterations", help="number of training iterations")
args = parser.parse_args()
config = vars(args)

epochs=10
if(config['iterations'] is not None):
  epochs = int(config['iterations'])


# Current Directory
dirname = os.path.dirname(__file__)
# Training Data seperated in folders with their classnames
data_path = pathlib.Path(os.path.join(dirname, 'data'))
# Stored Model
model_path = pathlib.Path(os.path.join(dirname, 'model'))
# The Names to classify by
class_names = os.listdir(data_path)
# Logs Directory
logdir = pathlib.Path(os.path.join(dirname, 'logs'))

#Training Configuration
batch_size = 32
img_height = 180
img_width = 180

def create_model():
  num_classes = len(class_names)

  # Data Augmentation to prevent overfitting
  data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",
                    input_shape=(img_height,
                                img_width,
                                3)),
    layers.RandomFlip("vertical",
                input_shape=(img_height,
                            img_width,
                            3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ])

  # Dropout to reduce overfitting
  model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
  ])

  #Compile Model
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  return model

def get_model():
  if(pathlib.Path.exists(model_path)):
    model = keras.models.load_model(model_path)
  else:
    model = create_model()

  return model

def show_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

#Amount of Images
image_count = len(list(data_path.glob('*/*.jpg')))
print(image_count)

# Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Tuning
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize
normalization_layer = layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

model = get_model()
model.summary()

#Training
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[tf.keras.callbacks.CSVLogger(logdir / "history.csv", append=True)],
)

model.save(model_path, save_format='tf')

show_plot(history)