# model_evaluation.py

import zipfile
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os 

# ---------- Load Images from ZIP ----------
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

# ---------- Evaluate Model ----------
def evaluate_model(zip_path):
    X, labels = load_images_from_zip(zip_path)
    X = X.reshape(-1, 48, 48, 1) / 255.0

    # Load the model
    model = load_model(os.path.join('model', 'expression_model.h5'))


    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    y = to_categorical(y)

    # Predict labels for validation data
    val_preds = model.predict(X, verbose=0)
    val_preds = np.argmax(val_preds, axis=1)
    y_true = np.argmax(y, axis=1)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, val_preds, encoder)

    # Plot accuracy and loss graphs
    plot_accuracy_loss_graph()

# ---------- Confusion Matrix ----------
def plot_confusion_matrix(y_true, y_pred, encoder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion Matrix - Validation Data")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ---------- Accuracy and Loss Graph ----------
def plot_accuracy_loss_graph():
    # Load history from the trained model
    history = np.load('history.npy', allow_pickle=True).item()  # Assuming history is saved in a file
    
    # Plot Train and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Train and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    zip_path = 'archive.zip'
    evaluate_model(zip_path)
