from datasets import Image
from flask import Flask, jsonify, make_response, request
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

model = load_model('keras_model.h5')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api', methods=['GET'])
def home():
    return make_response(jsonify(code='SUCCESS', message='Welcome!'), 200)


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return make_response(
            jsonify(code='BAD_REQUEST',
                    message='Tidak ada file gambar yang dikirimkan'), 400)

    file = request.files['file']

    if file.filename == '':
        return make_response(
            jsonify(code='BAD_REQUEST', message='File harus memiliki nama'),
            400)

    if allowed_file(file.filename):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        classes = ["early_bright", "healthy", "late_bright"]

        return make_response(
            jsonify(code='SUCCESS',
                    message='Prediksi berhasil',
                    data=classes[np.argmax(prediction)]), 200)


if __name__ == '__main__':
    app.run(debug=True)