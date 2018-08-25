import tensorflow as tf
import data


inputs = tf.placeholder(tf.float32, shape=(None, data.num_attributes))

class classifier:
    layers = [inputs]
    layers.append(tf.layers.dense(inputs=layers[-1], units=data.num_classes, activation=tf.nn.softmax))

# optimization-related
same_class_probabilities = tf.matmul(classifier.layers[-1], tf.transpose(classifier.layers[-1]))
log_same_class = tf.log(same_class_probabilities)
sameness = tf.placeholder(tf.float32, shape=(None, None))
loss = -tf.reduce_sum(tf.multiply(log_same_class, sameness))
optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)
