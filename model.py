import tensorflow as tf
import data


inputs = tf.placeholder(tf.float32, shape=(None, data.num_attributes))

class classifier:
    layers = [inputs]
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=layers[-1], units=data.num_classes, activation=tf.nn.softmax))

# maximize/minimize the probability that i and j are in the same class
# depending on what the expert says and with what certainty
same_class_probabilities = tf.matmul(classifier.layers[-1], tf.transpose(classifier.layers[-1]))
sameness = tf.placeholder(tf.float32, shape=(None, None)) # values of w from the original paper
loss = -tf.reduce_sum(tf.multiply(same_class_probabilities, sameness))
optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)
