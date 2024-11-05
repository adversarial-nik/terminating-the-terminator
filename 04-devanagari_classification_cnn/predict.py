from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

model = load_model('devnagri_model_1.3902e-05.net')
labels = [2,3,5,7]

# load dataset
dv = pd.read_csv('devanagari_prime_digits.csv', header=None).values

# predict
sample = dv[100,:-1].reshape(1,32,32,1)

pred = model.predict(sample)
class_index = np.argmax(pred)
print()
print(pred)
print('Prediction: ', labels[class_index])

# plot predicted sample
plt.imshow(sample.reshape(32,32))
plt.show()
