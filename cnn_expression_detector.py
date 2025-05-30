# cnn_expression_detector.py

import zipfile
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os

# ---------- Step 1: Load Images from ZIP ----------
def load_images_from_zip(zip_path):
    images, labels = [], []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.jpg') and 'images/validation/' in file:
                label = file.split('/')[-2]
                with zip_ref.open(file) as img_file:
                    img = Image.open(img_file).convert('L').resize((48, 48))
                    images.append(np.array(img))
                    labels.append(label)
    return np.array(images), labels

# ---------- Step 2: Train the CNN Model ----------
def train_model(zip_path):
    X, labels = load_images_from_zip(zip_path)
    X = X.reshape(-1, 48, 48, 1) / 255.0

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    y = to_categorical(y)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    model.save('expression_model.h5')
    print("✅ Model trained and saved as expression_model.h5")

    return encoder

# ---------- Step 3: Real-time Expression Detection ----------
def start_webcam_expression_detector(encoder):
    model = load_model('expression_model.h5')

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = grayscale[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = encoder.classes_[np.argmax(preds)]

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Expression Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Main Execution ----------
if __name__ == "__main__":
    zip_path = 'archive.zip'
    if not os.path.exists('expression_model.h5'):
        encoder = train_model(zip_path)
    else:
        # Load encoder manually (ensure order matches training labels)
        encoder = LabelEncoder()
        encoder.classes_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])  # adjust if needed

    start_webcam_expression_detector(encoder)
