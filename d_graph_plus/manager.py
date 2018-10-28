import numpy as np
import tensorflow as tf
from tqdm import trange
import d_graph_plus.optimizer as optimizer
from d_graph_plus.tasks import current as task
import d_graph_plus.data as data


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
history = {'objective': []}


def train(num_episodes=task.num_episodes):
    print('Training...')
    for episode in trange(num_episodes):
        examples, sameness = data.batch()
        _, objective = sess.run((optimizer.optimizer, optimizer.objective), feed_dict={task.model.input: examples, optimizer.sameness: sameness})
        history['objective'].append(objective)


def predict(examples):
    return sess.run(task.model.output, feed_dict={task.model.input: examples})


def classify(examples):
    predictions = predict(examples)
    return np.argmax(predictions, axis=-1)


def eval():
    examples, sameness = data.batch()
    return sess.run(optimizer.objective, feed_dict={task.model.input: examples, optimizer.sameness: sameness})


def correlation_matrix():
    output = predict(task.examples)
    result = []
    for class_id in range(task.num_classes):
        class_correlation = np.mean(output[task.classes == class_id], axis=0)
        result.append(class_correlation)
    return np.array(result)
