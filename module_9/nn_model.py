import tensorflow as tf
from tensorflow import keras
from keras import datasets
import os

import numpy as np
import matplotlib.pyplot as plt

tf.keras.datasets.fashion_mnist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

print(train_images[0, 23, 23])

print(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.show()

##data preprocessing
## apply some transformations before feeding to model.
## this case, we divide by 255 , we divide all values by 255 to be between 0 and 1

train_images = train_images / 255
test_images = test_images / 255

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])

#layer 1 input layer 784 neurons
#layer 2 dense, layer fully conneted and each neuron from the previous connects to each neuron of this layer.
#128 neurons
#layer 3 output layer, 10 neurons, 10 classes
#activation functionsj
#softmax makes sure all output neurons add up to 1.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

model.save('saved_model/my_model')

print('Test accuracy:', test_acc)

#NN with train data duoda 91 proc accuracy.
#bet su testing data duoda 88, cia yra overfitting. nes training data epochom naudojo tapacia data ir pradejo isiminti

predictions = model.predict(test_images)

print(class_names[np.argmax(predictions[5])])

plt.figure()
plt.imshow(test_images[5])
plt.colorbar()
plt.show()