import numpy as np
import tensorflow as tf
from flask import Flask
from flask_limiter import Limiter
from flask import Blueprint, request, jsonify
from flask_limiter.util import get_remote_address
from keras.preprocessing import image as image_utils

model=tf.keras.models.load_model(r'C:\xampp\htdocs\api-v1\test0orl\Oral_cancer.h5')            # ==> model we want predict path


image_path = r"C:\xampp\htdocs\api-v1\test0orl\00.jpg"
image = image_utils.load_img(image_path)
input_arr = image_utils.img_to_array(image)
scaled_img = np.expand_dims(input_arr, axis=0)


class_names = ['CANCER', 'NON CANCER']
pred = model.predict(scaled_img)
output = class_names[np.argmax(pred)]
print(output)
