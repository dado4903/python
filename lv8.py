import shutil
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import pandas as pd
from pathlib import Path
from keras import Sequential
from tensorflow.python.keras.layers.core import Dropout

train_ds = image_dataset_from_directory(
    directory='Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48))



num_classes = 43
input_shape = (48, 48, 3)

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])
model.summary()
# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy',
                optimizer='Nadam',
                metrics=['accuracy'])

# TODO: provedi ucenje mreze
model.fit(train_ds, epochs=10, batch_size=32)

test_ds = image_dataset_from_directory(
    directory='Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48))

loss_and_metrics = model.evaluate(test_ds, batch_size=128)
print(loss_and_metrics)