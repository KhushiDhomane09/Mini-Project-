from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import firebase_admin
from firebase_admin import credentials, auth, firestore
import base64
from io import BytesIO

# ========== Flask and Firebase Setup ==========
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ========== Load CNN Model ==========
model = load_model('model/expression_model.h5')
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ========== Firebase Session Login ==========
@app.route('/sessionLogin', methods=['POST'])
def session_login():
    data = request.get_json()
    id_token = data.get('idToken')

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        session['user'] = uid
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 401

# ========== Predict from Webcam Frame ==========
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        print("❌ Unauthorized: No user session.")
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    data = request.get_json()
    base64_image = data.get('image')

    if not base64_image:
        print("❌ No image data received.")
        return jsonify({'success': False, 'message': 'No image data received'}), 400

    try:
        # Decode and preprocess image
        img_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(img_data)).convert('L')  # Grayscale
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # Debug image saving (optional)
        # img.save("last_received_frame.jpg")

        # Predict
        prediction = model.predict(img_array)
        predicted_label = labels[np.argmax(prediction)]
        print("✅ Prediction:", predicted_label)

        return jsonify({'success': True, 'expression': predicted_label})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({'success': False, 'message': 'Error during prediction'}), 500


    
# ========== Upload Image (ImageToText Section) ==========
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return redirect(url_for('login_signin'))

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    try:
        image = Image.open(file.stream).convert('L')
        image = image.resize((48, 48))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        prediction = model.predict(img_array)
        predicted_label = labels[np.argmax(prediction)]

        db.collection('image_to_text_data').add({
            'user_id': session['user'],
            'expression': predicted_label,
            'source': 'image',
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({'success': True, 'expression': predicted_label})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ========== Routes ==========
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_signin'))

@app.route('/login', methods=['GET'])
def login_signin():
    return render_template('login-signin.html')

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('index.html')

@app.route('/capture')
def capture():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('CaptureNMF.html')

@app.route('/imagetoText')
def imagetotext():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('ImageToText.html')

@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('AboutUs.html')

@app.route('/team')
def team():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('team.html')

@app.route('/contact')
def contact():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('Contact.html')

@app.route('/feedback')
def feedback():
    if 'user' not in session:
        return redirect(url_for('login_signin'))
    return render_template('feedback.html')

# ========== Main ==========
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
