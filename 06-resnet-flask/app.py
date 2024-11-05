import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Load ResNet50 model
model = ResNet50(weights='imagenet')

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def model_predict(img_path):
    """Load and preprocess image, predict using ResNet50."""
    img = Image.open(img_path).resize((224, 224))  # ResNet50 expects 224x224 input
    img = np.array(img)  # Convert to NumPy array
    if img.shape[2] == 4:  # Check if RGBA, convert to RGB
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for ResNet50

    preds = model.predict(img)  # Predict the image
    decoded_preds = decode_predictions(preds, top=5)[0]  # Decode top 5 predictions

    return decoded_preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('predict', filename=filename))

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preds = model_predict(file_path)

    return render_template('result.html', image_file=filename, predictions=preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
