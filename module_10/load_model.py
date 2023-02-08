
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('saved_model/my_model')

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))
#     plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)