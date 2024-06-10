from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__,static_url_path='/static')

# Load your trained model
model = load_model('gbgclassifier1.h5')

# Predefined metrics (Replace these with your actual metrics)
MODEL_ACCURACY = 0.94
MODEL_PRECISION = 0.75
MODEL_RECALL = 0.73

# Preprocessing function
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image/255.0, axis=0)
   
    return image

@app.route('/')
def index():
    return render_template('index.html', accuracy=MODEL_ACCURACY, precision=MODEL_PRECISION, recall=MODEL_RECALL)

@app.route('/predict', methods=['POST'])
def predict():
    class_dict = {
        0: 'Battery',
        1: 'Biological',
        2: 'Cardboard',
        3: 'Clothes',
        4: 'Glass',
        5: 'Metal',
        6: 'Paper',
        7: 'Plastic',
        8: 'Shoes',
        9: 'Trash'
    }
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image, target_size=(224, 224))
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_dict.get(predicted_class_index)
            if predicted_class is None:
                return jsonify({'error': 'Predicted class index not found'}), 500
            return jsonify({'predicted_class': predicted_class}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
