import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import h5py
import random
import cv2
from tqdm import tqdm

x = []
y = []

f = open('x50.npy', 'rb')
x = np.load(f)
f.close()
f = open('y50.npy', 'rb')
y = np.load(f)
f.close()

model = Sequential()
model.add(Dense(16, input_shape=x.shape[1:], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.add(Flatten())


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=5, epochs=20, validation_split=0.2)
model.predict(x)
model.save('model2.h5')
