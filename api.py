from flask import Flask, request, jsonify,render_template
import os
from PIL import Image
from io import BytesIO
import json
from img2vec_pytorch import Img2Vec
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
model = load_model('models/model.h5')

# Initialize the feature extractor
img2vec = Img2Vec(cuda=False)

# Load the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/', methods=['GET','POST'])
def index():
    data ={
        "isPredicted" : False
    }

    if request.method == 'GET':
        return render_template('index.html',data=data)
    
    else:
        file = request.files['image']

        # Open the image file
        img = Image.open(BytesIO(file.read()))

        # Preprocess the image
        resized_img = img.resize((224, 224))

        # Extract features
        vec = img2vec.get_vec(img, tensor=True)
        arr = vec.numpy()
        arr = arr.reshape(1, 512)

        # Make a prediction
        output = model.predict(arr)[0]
        prediction = np.argmax(output)
        probabilities = {class_names[str(i)]: "{:.2f}%".format(prob * 100) for i, prob in enumerate(output)}

        # Return the prediction and probabilities
        # return jsonify({
        #     'prediction': class_names[str(prediction)],
        #     'probabilities': probabilities
        # })
        data['isPredicted'] = True
        data['prediction'] = class_names[str(prediction)]
        data['probabilities'] = probabilities
        
        return render_template('index.html',data=data)


    
@app.route('/<family>', methods=['GET'])
def show(family):
    families=None
    with open('./familyData/families.json', 'r', encoding='utf-8') as f:
        families = json.load(f)

    # Check if the requested family exists in the loaded data
    if family.upper() in families:
        return render_template('family.html',data=families[family.upper()])
    else:
        return jsonify({'error': 'Family not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)