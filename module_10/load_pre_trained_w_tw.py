import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras import datasets, layers, models
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]' ],
        with_info=True,
        as_supervised=True
)

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# for img, label in raw_train.take(2):
#     print("Original shape", img.shape)

# for img, label in train.take(2):
#     print("Original shape", img.shape)

#     

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model('saved_model/loaded_model_w_tweaks')

#predictions = model.predict(test, batch_size=BATCH_SIZE)

#print(predictions[1])

plt.figure()
plt.imshow(raw_test[0][0])
plt.show()

reshaped = tf.expand_dims(test[0], axis=0).shape.as_list()
print(reshaped.shape)