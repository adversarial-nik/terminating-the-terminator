import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
from PIL import Image

ip_img_path = "elephant.jpg"
hacked_img_path = "/tmp/hacked_ele.png"
diff_img_path = "/tmp/diff.png"
target_class = 47

# load model
model = inception_v3.InceptionV3()
ip_layer = model.layers[0].input
op_layer = model.layers[-1].output


# Load the image to hack
img = image.load_img(ip_img_path, target_size=(299, 299))
ip_image = image.img_to_array(img)
ip_image /= 255.
ip_image -= 0.5
ip_image *= 2.

# Add a 4th dimension for batch size (as Keras expects)
ip_image = np.expand_dims(ip_image, axis=0)

# set bounds for change during optimization
max_change_above = ip_image + 5.0
max_change_below = ip_image - 5.0


# Create a copy of the input image to hack on
hacked_image = np.copy(ip_image)


cost_function = op_layer[0, target_class]

gradient_function = K.gradients(cost_function, ip_layer)[0]
grab_cost_and_gradients_from_model = K.function([ip_layer, K.learning_phase()], [cost_function, gradient_function])
cost = 0.0
learning_rate = 0.9

while cost < 0.98:
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    
    # Move the hacked image one step further towards fooling the model 
    hacked_image += gradients * learning_rate

    hacked_image = np.clip(hacked_image, -1.0, 1.0)

    print("\033[92m [+] Predicted probability of target class: {:.8}\033[0m".format(cost))

hacked_img = hacked_image[0]
hacked_img /= 2.
hacked_img += 0.5
hacked_img *= 255.

im = image.array_to_img(hacked_img)
im.save(hacked_img_path)


diff_img = hacked_image[0] - ip_image[0]
diff_img /= 2.
diff_img += 0.5
diff_img *= 255.

diff_im = Image.fromarray(diff_img.astype(np.uint8))
diff_im.save(diff_img_path)
