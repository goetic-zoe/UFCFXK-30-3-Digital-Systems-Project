# Skin Cancer Detection Flask Application

## Overview

This project is a Flask application designed to detect different types of skin cancer from images. The application uses a pre-trained convolutional neural network (CNN) model that has been trained on the ISIC dataset. The user can upload an image, and the app will predict the type of skin lesion present in the image.

## Usage

### Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.12.13 installed and a virtual environment set up:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   This will install all the necessary packages including Flask, TensorFlow, Keras, etc.

### Running the Application

1. **Start the Flask Server:**

   ```bash
   python app.py
   ```

2. **Access the Application:**

   Open a web browser and go to `http://127.0.0.1:5000/`. You will see the main dashboard of the application.

3. **Upload an Image for Prediction:**

    - Click on the "Choose File" button to select an image.
    - Click on the "Predict" button to upload the image and get a prediction.

4. **View Results:**

   The application will display the predicted class of the skin lesion along with the confidence score. It will also show the uploaded image for reference.

## Explanation of Key Files

### `app.py`

This file contains the Flask application code. It defines two routes:

- `/`: The main dashboard route where users can upload images.
- `/predict`: The prediction route that processes the uploaded image, makes a prediction using the trained model, and returns the results to the user.

### `train.py`

This script is used to train the CNN model on the skin cancer dataset. It includes data augmentation, model architecture definition, training process, and visualization of training metrics.

- **Data Augmentation:** The script uses the `Augmentor` library to augment the training images.
- **Model Architecture:** A convolutional neural network with 4 blocks is defined using TensorFlow/Keras.
- **Training:** The model is trained for a specified number of epochs, with callbacks for saving checkpoints and reducing learning rate based on validation loss.

### `Skin_Cancer_Detection_CNN.ipynb`

This Jupyter Notebook contains the initial steps for training the CNN model. It includes data loading, visualization, model definition, and training process. The notebook also provides an example prediction using a test image.

## Notes

- **Model File:** Ensure that the trained model file (`skin_CNN2v3.keras`) is located in the `models` directory before running the Flask application.
- **Dataset:** The dataset used for training should be available in the `dataset` directory. If not, you may need to download it and place it in the appropriate subdirectories. However, it will attempt to automatically download it when training.
