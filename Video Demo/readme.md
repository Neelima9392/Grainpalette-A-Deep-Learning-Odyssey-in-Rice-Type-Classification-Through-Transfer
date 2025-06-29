video demonstration of logic
The GrainPalette project successfully demonstrates the power and potential of deep learning and transfer learning in the field of agriculture. By leveraging Convolutional Neural Networks (CNN) and the MobileNetV4 architecture, the system can accurately classify rice grain types from uploaded images with minimal user input.

This AI-driven tool is not just a technological demonstration—it offers real-world value for:

Farmers, by enabling better crop planning and input management based on rice type,

Agricultural scientists and extension workers, for rapid field-level identification and data collection,

Home growers and educators, for fostering awareness of crop diversity and promoting smart agriculture practices.

The project also showcases the end-to-end development of a machine learning application, from dataset preparation, image preprocessing, model training, and evaluation to building a fully functional Flask web application.

Overall, GrainPalette stands as a practical, scalable, and educational application of AI in agriculture, reflecting how data-driven decision-making can lead to more sustainable and efficient farming practices.
User Uploads Image via the web interface.

Image Preprocessing:

Resize to 224×224

Normalize pixel values

Convert to array format

Load Pre-trained Model (rice_model.keras)

Model Prediction:

Pass preprocessed image to the model

Get predicted rice type

Return Result to the user via the HTML interface.

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice_model.keras')

class_labels = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Other']  # example classes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_path = os.path.join('uploads', img.filename)
    img.save(img_path)

    img_loaded = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_loaded)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = class_labels[np.argmax(prediction)]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
