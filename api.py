from flask import Flask, request, jsonify
from waitress import serve
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
reader = easyocr.Reader(['ar', 'en'])


def preprocess_image(image_bytes):
    # image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # image_np = np.array(image)
    # gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return thresh
    '''new code'''
    # image_file = io.BytesIO(image_bytes)
    # img_str = image_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    
    kernal = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    return image


@app.route("/")
def root():
    """Site main page handler function."""

    try:
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(template_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading template: {e}")
        return f"Error loading page: {str(e)}", 500


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    processed_image = preprocess_image(image_bytes)

    results = reader.readtext(processed_image)
    extracted_text = ' '.join([(res[1] + '\n') for res in results])

    return jsonify({'text': extracted_text})


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)