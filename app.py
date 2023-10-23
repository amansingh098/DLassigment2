from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename, safe_join
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os



# Path to the model
MODEL_PATH = 'model.h5'

app = Flask(__name__)

# Load the Keras model
model = load_model(MODEL_PATH)

def preprocess_new_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    
    # Convert the image to a numpy array and rescale the pixel values
    x = image.img_to_array(img) / 255.0
    
    # Expand the dimensions to match the shape the model expects
    x = np.expand_dims(x, axis=0)
    
    return x

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = safe_join(app.root_path, 'static/uploads', file.filename)
            file.save(file_path)
            
            # Predict the class
            img = preprocess_new_image(file_path)
            prediction = model.predict(img)
            class_names = ['glioma', 'meningioma', 'noutumor', 'pituitary']
            predicted_class = class_names[np.argmax(prediction)]
            return render_template('index.html', prediction=predicted_class, uploaded_image=file_path)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
