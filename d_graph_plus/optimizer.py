import tensorflow as tf
from d_graph_plus.tasks import current as task


# maximize/minimize the probability that i and j are in the same class
# depending on what the expert says and with what certainty
same_class_probabilities = tf.matmul(task.model.output, tf.transpose(task.model.output))
sameness = tf.placeholder(tf.float32, shape=(None, None)) # values of w from the original paper
loss = -tf.reduce_sum(tf.multiply(same_class_probabilities, sameness))
optimizer = tf.train.AdamOptimizer(task.learning_rate).minimize(loss)
