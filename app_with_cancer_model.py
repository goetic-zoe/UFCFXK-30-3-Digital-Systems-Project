from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model (make sure the path is correct)
model = tf.keras.models.load_model('models/trained_cancer_cnn.keras')

@app.route('/')
def dashboard():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['image_file']
    if file.filename == '':
        return redirect(request.url)
    
    # Process the image
    img = Image.open(io.BytesIO(file.read())).resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 
                   'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 
                   'squamous cell carcinoma', 'vascular lesion']
    predicted_class = class_names[predicted_class_index]
    
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
