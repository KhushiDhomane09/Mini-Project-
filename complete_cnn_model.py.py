# expression_detector.py

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Load the trained model
model_path = os.path.join('model', 'expression_model.h5')
model = load_model(model_path)

def start_webcam_expression_detector(encoder):
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
            roi = np.expand_dims(roi, axis=-1)     # Shape: (48, 48, 1)
            roi = np.expand_dims(roi, axis=0)      # Shape: (1, 48, 48, 1)

            preds = model.predict(roi, verbose=0)[0]
            label = encoder.classes_[np.argmax(preds)]

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Expression Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

    start_webcam_expression_detector(encoder)
