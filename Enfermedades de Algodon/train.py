import os
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = "cotton_data/"
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
train_counts = Counter(train_gen.classes)
val_counts = Counter(val_gen.classes)

df_stats = pd.DataFrame({
    'Clase': class_names,
    'Train': [train_counts[i] for i in range(len(class_names))],
    'Val': [val_counts[i] for i in range(len(class_names))]
})

print("\nEstadísticas por clase (antes del entrenamiento):")
print(df_stats.to_string(index=False))

model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

model.save("modelo_algodon.h5")
print("\nModelo guardado como 'modelo_algodon.h5'")