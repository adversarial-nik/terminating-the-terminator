import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras import backend as K
from PIL import Image
import os
import sys

model = inception_v3.InceptionV3()

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    original_image = image.img_to_array(img)
    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.
    x = np.expand_dims(original_image, axis=0)
    pred = model.predict(x)
    print(f'\n\n')
    print('\033[92m Model Predictions: '+str(decode_predictions(pred, top=3))+'\033[0m')
    
predict_img(sys.argv[1])
