from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model (make sure the path is correct)
model = tf.keras.models.load_model('models/trained_cancer_cnn.keras')


@app.route('/')
def dashboard():
    return render_template('index.html', prediction=None, image_filename='placeholder.jpg')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return redirect(request.url)

    file = request.files['image_file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    image_filename = file.filename
    file.save(str(os.path.join(app.config['UPLOAD_FOLDER'], str(image_filename))))

    image = tf.io.read_file(os.path.join(app.config['UPLOAD_FOLDER'], str(image_filename)))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image_size = (180,180)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    image_arr = tf.keras.utils.img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)

    # Make prediction
    prediction = model.predict(image_arr)
    predicted_class_index = np.argmax(prediction[0])
    class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma',
                   'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
                   'squamous cell carcinoma', 'vascular lesion']
    predicted_class = class_names[predicted_class_index]

    return render_template('index.html', prediction=predicted_class, image_filename=image_filename)


if __name__ == '__main__':
    app.run(debug=True)