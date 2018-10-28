import tensorflow as tf
from d_graph_plus.tasks import current as task


# maximize/minimize the probability that i and j are in the same class
# depending on what the expert says and with what certainty
posterior_probabilities = tf.matmul(task.model.output, tf.transpose(task.model.output))
sameness = tf.placeholder(tf.float32, shape=(None, None)) # values of l or w from the original paper
probability_objective = tf.reduce_sum(tf.multiply(posterior_probabilities, sameness))
l2_loss = tf.losses.get_regularization_loss()
objective = probability_objective - l2_loss # equation (7) in original paper
optimizer = tf.train.AdamOptimizer(task.learning_rate).minimize(-objective)
