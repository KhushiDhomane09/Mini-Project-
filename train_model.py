# train_model.py

from extract_dataset import load_images_from_zip
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# ===== Load Data =====
X, labels = load_images_from_zip('archive.zip')
X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0

# ===== Encode Labels =====
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)

# ===== Build CNN Model =====
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== Callbacks =====
checkpoint = ModelCheckpoint('model/expression_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ===== Train the Model =====
model.fit(
    X, y,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop],
    verbose=2
)
