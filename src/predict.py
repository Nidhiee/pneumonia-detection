import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 150
model = tf.keras.models.load_model("models/pneumonia_model.h5")

def predict_xray(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Pneumonia Detected "
    else:
        return "Normal X-ray "

# Example
result = predict_xray("sample_xray.jpeg")
print("Result:", result)
