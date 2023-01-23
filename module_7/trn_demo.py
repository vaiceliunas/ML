from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import tensorflow as tf

import pandas as pd
import logging
logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def input_fn(features, labels, training=True, batch_size=256):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(100).repeat()
    
    return dataset.batch(batch_size)

CSV_COLUMN_NAMES = ['SepalLenhth', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file( "iris.training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file('iris_test.csv', "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

#print(train.shape)
logging.info('I am info')

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    

config = tf.estimator.RunConfig().replace(keep_checkpoint_max = 5, 
                    log_step_count_steps=20, save_checkpoints_steps=200)
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3, config = config) 

train_input_fn_obj = lambda: input_fn(train, train_y, training=True)
test_input_fn_obj = lambda: input_fn(test, test_y, training=True)

classifier.train(lambda: input_fn(train, train_y, training=True), steps=5000)
print("evaluatinsim")
classifier.evaluate(test_input_fn_obj)