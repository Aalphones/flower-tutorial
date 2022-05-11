import numpy as np
import os
import shutil
import tensorflow as tf
import argparse

from tensorflow import keras

import pathlib

# Parse arguments
parser = argparse.ArgumentParser(description="Training CLI",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-out", "--output", help="Output directory")
parser.add_argument("-in", "--input", help="Input directory")
parser.add_argument("-search", "--search", help="Search By")
args = parser.parse_args()
config = vars(args)

img_height = 180
img_width = 180
dirname = os.path.dirname(__file__)

model_path = pathlib.Path(os.path.join(dirname, 'model'))
data_path = pathlib.Path(os.path.join(dirname, 'data'))
input_path = pathlib.Path(os.path.join(dirname, 'input'))
if(config['input'] is not None):
  input_path = config['input']

output_path = pathlib.Path(os.path.join(dirname, 'output'))
if(config['output'] is not None):
  output_path = config['output']

if(config['search'] is not None):
  desired_class = config['search']

def move_file(result, file, confidence):
  if(config['search'] is None):
    resulting_path = os.path.join(output_path, result)
  elif(desired_class != result):
    resulting_path = os.path.join(output_path, "NOT_OK")
  else:
    resulting_path = os.path.join(output_path, result)

  resulting_name = "{:.2f}_{}".format(confidence, name)
  target = os.path.join(resulting_path, resulting_name)

  if not os.path.exists(resulting_path):
    os.makedirs(resulting_path)

  shutil.copy(file, target)

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
  confidence = 100 * np.max(score)

  move_file(result, file, confidence)
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(result, confidence)
  )

# Retrieve
model = keras.models.load_model(model_path)
class_names = os.listdir(data_path)
to_be_checked = os.listdir(input_path)

for name in to_be_checked:
  predict_image(name)


