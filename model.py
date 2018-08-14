import tensorflow as tf
import data


inputs = tf.placeholder(tf.float32, shape=(None, data.num_attributes))

class classifier:
    layers = []
    layers.append(tf.layers.dense(inputs=inputs, units=4, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=inputs, units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=inputs, units=4, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=inputs, units=data.num_classes, activation=tf.nn.softmax))

# optimization-related
same_class_probabilities = tf.matmul(classifier.layers[-1], tf.transpose(classifier.layers[-1]))
log_same_class = tf.log(same_class_probabilities)
sameness = tf.placeholder(tf.float32, shape=(None, None))
loss = tf.reduce_mean(tf.multiply(log_same_class, sameness))
optimizer = tf.train.AdamOptimizer(.01).minimize(loss)
