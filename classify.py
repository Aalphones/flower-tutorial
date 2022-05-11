import numpy as np
import os
import shutil
import tensorflow as tf

from tensorflow import keras

import pathlib

img_height = 180
img_width = 180
dirname = os.path.dirname(__file__)

model_path = pathlib.Path(os.path.join(dirname, 'model'))
data_path = pathlib.Path(os.path.join(dirname, 'data'))
input_path = pathlib.Path(os.path.join(dirname, 'input'))
output_path = pathlib.Path(os.path.join(dirname, 'output'))

def move_file(result, file):
  resulting_path = os.path.join(output_path, result)
  target = os.path.join(resulting_path, name)

  if not os.path.exists(resulting_path):
    os.makedirs(resulting_path)

  print (target)
  shutil.move(file, target)

def predict_image(name):
  file = os.path.join(input_path, name)

  img = tf.keras.utils.load_img(
      file, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  result = class_names[np.argmax(score)]
 
  move_file(result, file)
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(result, 100 * np.max(score))
  )

# Retrieve
model = keras.models.load_model(model_path)
class_names = os.listdir(data_path)
to_be_checked = os.listdir(input_path)

for name in to_be_checked:
  predict_image(name)


