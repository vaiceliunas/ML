import tensorflow as tf
from tensorflow import keras
from keras import datasets
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('saved_model/my_model')

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#predictions = new_model.predict(test_images)


#predictions = new_model.predict(test_images)

#pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')

#np.savetxt("score_2.csv", predictions, delimiter=",")

predictions = pd.read_csv('score_2.csv').to_numpy()

while True:
    try:
        integer = int(input("Enter an integer 0 to 10000\n"))
    except ValueError:
        print("Value was not an integer")
        break
    else:
        print("Should be")
        print(class_names[test_labels[integer]])
        print("Class ID")
        print(test_labels[integer])
        print("---")
        print("Prediction:")
        print(class_names[np.argmax(predictions[integer-1])])
        print("Class ID")
        print(np.argmax(predictions[integer-1]))

        plt.figure()
        plt.imshow(test_images[integer])
        plt.colorbar()
        plt.show()

# while True:
#     try:
#         integer = int(input("Input prediction\n"))
#     except ValueError:
#         print("Value was not an integer")
#         break
#     else:
#         print(predictions[integer])
#         print(predictions2[integer-1])




