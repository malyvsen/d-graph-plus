import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# dataset
image_dim = 28
num_classes = 10
num_kept = 4096

(examples, classes), (_, _) = mnist.load_data()
examples, classes = examples[:num_kept], classes[:num_kept] # they are already shuffled, no worries
examples = examples / 255.0


# hyperparameters
num_must = 1024
num_cannot = 1024
batch_size = 128
learning_rate = 4e-4
num_episodes = 1024


class model:
    input = tf.placeholder(tf.float32, shape=(None, image_dim, image_dim))
    flat = tf.layers.Flatten()(input)
    weights = tf.Variable(tf.truncated_normal(shape=(image_dim * image_dim, num_classes)))
    biases = tf.Variable(tf.truncated_normal(shape=(num_classes,)))
    output = tf.nn.softmax(tf.matmul(flat, weights) + biases)
