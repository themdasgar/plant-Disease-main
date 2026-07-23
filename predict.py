import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def cnn_predict(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    confidence = np.max(prediction)

    class_index = np.argmax(prediction)

    label = class_names[class_index]

    plant, disease = label.split("___")

    return plant, disease, confidence
# Test
