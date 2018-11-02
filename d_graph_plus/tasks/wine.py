import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine


# dataset
num_attributes = 13
num_classes = 3

data = load_wine()
examples = data.data
classes = data.target


# hyperparameters
num_must = 0
num_cannot = 0
neighborhood_size = 4
auxiliary_weight = 1
gamma = 1 # parameter for RBF
regularizer = tf.contrib.layers.l2_regularizer(0.0)
batch_size = len(examples) # examples
min_batch_must = 0 # pairs of examples
min_batch_cannot = 0 # pairs of examples
learning_rate = 1e-3
num_episodes = 64


class model:
    input = tf.placeholder(tf.float32, shape=(None, num_attributes))
    layers = [input]
    output = tf.layers.dense(inputs=layers[-1], units=num_classes, activation=tf.nn.softmax, kernel_regularizer=regularizer)
    layers.append(output)
