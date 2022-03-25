from datasets import Image
from flask import Flask, jsonify, make_response, request
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing import image

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

model = load_model('model.h5')

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
        image_pil = Image.open(file).resize((64, 64))
        test_image = image.img_to_array(image_pil)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'early_blight'
        elif result[0][1] == 1:
            prediction = 'healthy'
        else:
            prediction = 'late_blight'

        return make_response(
            jsonify(code='SUCCESS',
                    message='Prediksi berhasil',
                    data=prediction), 200)


if __name__ == '__main__':
    app.run(debug=True)