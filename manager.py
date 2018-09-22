import numpy as np
import tensorflow as tf
from tqdm import trange
import optimizer
from tasks import current as task
import data


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def train(num_episodes=4096):
    print('Training...')
    for episode in trange(num_episodes):
        examples, sameness = data.batch(size=task.batch_size)
        sess.run(optimizer.optimizer, feed_dict={task.model.input: examples, optimizer.sameness: sameness})


def predict(examples):
    return sess.run(task.model.output, feed_dict={task.model.input: examples})


def classify(examples):
    predictions = predict(examples)
    return np.argmax(predictions, axis=-1)


def eval():
    examples, sameness = data.batch()
    return sess.run(optimizer.loss, feed_dict={task.model.input: examples, optimizer.sameness: sameness})
