
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, inception_v3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Replace with your image path
img_path = 'street_sign.jpg'  

# Disable eager execution and GPU (as in the example)
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    return x

def predict(images):
    return model.predict(images)

# Load and preprocess the image
processed_image = preprocess_image(img_path)

# Get model predictions
preds = predict(processed_image)
decoded_preds = decode_predictions(preds)[0]

print("Top 5 predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")

# Create LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate explanation
explanation = explainer.explain_instance(processed_image[0].astype('double'), 
                                         predict, 
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000)

# Visualize explanation for the top predicted label
top_label = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(processed_image[0] / 2 + 0.5)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title(f"LIME Explanation for '{decoded_preds[0][1]}'")
plt.axis('off')

plt.tight_layout()
plt.show()
