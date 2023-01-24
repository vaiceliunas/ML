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
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

def input_fn2(features, batch_size=256):

    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
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
test_input_fn_obj = lambda: input_fn(test, test_y, training=False)

classifier.train(lambda: input_fn(train, train_y, training=True), steps=5000)
print("evaluatinsim")
eval_result = classifier.evaluate(test_input_fn_obj)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as promted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn2(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))