import numpy as np
import tensorflow as tf
import os


# dataset
num_attributes = 16
num_classes = 26

examples = []
classes = []
with open(os.path.join(os.path.dirname(__file__), 'letter.data')) as data_file:
    for line in data_file:
        if len(examples) >= 1024:
            break # no need for so many examples
        line_split = line.split(',')
        classes.append(ord(line_split[0]) - ord('A'))
        examples.append(np.array(line_split[1:], dtype=np.float32))

examples = np.array(examples)
classes = np.array(classes)


# hyperparameters
num_must = 128
num_cannot = 128
neighborhood_size = 4
auxiliary_weight = 1
gamma = 1 # parameter for RBF
regularizer = tf.contrib.layers.l2_regularizer(0.0)
batch_size = len(examples) # examples
min_batch_must = 0 # pairs of examples
min_batch_cannot = 0 # pairs of examples
learning_rate = 1e-3
num_episodes = 256


class model:
    input = tf.placeholder(tf.float32, shape=(None, num_attributes))
    layers = [input]
    output = tf.layers.dense(inputs=layers[-1], units=num_classes, activation=tf.nn.softmax, kernel_regularizer=regularizer)
    layers.append(output)
