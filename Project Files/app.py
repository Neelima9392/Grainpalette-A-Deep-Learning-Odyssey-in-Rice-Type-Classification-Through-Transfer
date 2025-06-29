from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("rice_model.keras")  # Make sure this file is in the same folder

# Mapping class indices to names
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded image to a temporary location
    img_path = os.path.join('static', 'upload.jpg')
    file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]
    confidence = float(np.max(prediction)) * 100

    print("Rendering template with:", predicted_class, round(confidence, 2), img_path)

    return render_template('results.html',
                           prediction=predicted_class,
                           confidence=round(confidence, 2),
                           image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)