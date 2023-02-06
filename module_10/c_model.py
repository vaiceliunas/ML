import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

IMG_INDEX = 2

plt.figure()
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

model = models.Sequential()

#32 filters of 3x3 over input data
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#pooling of 2x2 to reduce dimensionality
model.add(layers.MaxPooling2D((2, 2)))
#64 filters of 3x3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#nera padding, todel 30x 30 (tiek kartu paimam filtra), 32 filtrai
#  conv2d (Conv2D)             (None, 30, 30, 32)        896

#  max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
#  )

#  conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496

#  max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
#  2D)

#  conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
#cia isprintinam i 1d, perduodam dense layeriui, ir galiausiai output layeriui

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

model.save('saved_model/my_model')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)