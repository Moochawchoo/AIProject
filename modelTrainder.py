import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
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
x = x/255

model = Sequential()
model.add(Conv2D(5, (3,3), input_shape = x.shape[1:]))#edit first value of Conv2d if having memory problems make it lower
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(5, (3,3)))#edit first value of Conv2d if having memory problems make it lower
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(5))#You can also lower this value if having memory problems
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=5, epochs=10,  validation_split=0.2)#lower batch size if having memory problems
