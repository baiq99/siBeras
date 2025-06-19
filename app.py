import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
from io import BytesIO
from base64 import b64decode

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load('model/svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')
class_labels = model.classes_.tolist()

def adjust_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def convertToHSV_withMask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 20), (180, 255, 255))
    return cv2.bitwise_and(hsv, hsv, mask=mask)

def extract_color_moments(image_hsv_masked):
    features = []
    for channel in cv2.split(image_hsv_masked):
        channel = channel.flatten().astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std**3 + 1e-10)
        features.extend([mean, std, skewness])
    return features

def fullPreprocessingHu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def extract_hu_moments(processed_image, min_area_threshold=50, top_n=3):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
    if not contours:
        return [0.0] * 7
    contour_areas = [cv2.contourArea(c) for c in contours]
    mean_area = np.mean(contour_areas)
    closest = sorted(contours, key=lambda c: abs(cv2.contourArea(c) - mean_area))[:top_n]
    hu_features = []
    for c in closest:
        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        hu_features.append(hu_log)
    return np.mean(hu_features, axis=0).tolist()

def predict_from_image(image):
    image = cv2.resize(image, (500, 500))
    image = adjust_background(image)
    hsv_masked = convertToHSV_withMask(image)
    color_features = extract_color_moments(hsv_masked)
    hu_input = fullPreprocessingHu(image)
    hu_features = extract_hu_moments(hu_input)
    final_features = np.array(color_features + hu_features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = np.max(probabilities) * 100
    prob_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(class_labels, probabilities)}
    return prediction, confidence, prob_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    camera_data = request.form.get('cameraImage')

    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)

    elif camera_data:
        try:
            img_data = b64decode(camera_data.split(',')[1])
            pil_image = PILImage.open(BytesIO(img_data)).convert('RGB')
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            filename = "capture.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, image)
        except Exception as e:
            return f"Gagal membaca gambar dari kamera: {e}", 400

    else:
        return 'No file selected or captured.', 400

    prediction, confidence, prob_dict = predict_from_image(image)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=f"{confidence:.2f}%",
                           image_path=filepath,
                           prob_dict=prob_dict)

# Tidak ada app.run di sini karena akan dijalankan oleh Gunicorn
