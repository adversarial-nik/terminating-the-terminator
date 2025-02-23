# 12 minutes
# Here is a simpler example of the use of LIME for image classification by using Keras (v2 or greater)

import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np

# set image path to generate the explanation for a label
label = 919 # replace with label number from inception class labels
image_path = './street_sign.jpg'

# # Using Inception
# Here we create a standard InceptionV3 pretrained model and use it on images by first preprocessing them with the preprocessing tools
inet_model = inc_net.InceptionV3()


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


# ## Let's see the top 5 prediction for some image
images = transform_img_fn([image_path])
plt.imshow(images[0] / 2 + 0.5)
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)


# ## Explanation
# Now let's get an explanation
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)


from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()
