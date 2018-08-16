import numpy as np
import tensorflow as tf
import model
import data


# training hyperparameters
batch_size = 100
constraint_ratio = .25


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def train(num_episodes=1024):
    for episode in range(num_episodes):
        examples, sameness = data.batch(size=batch_size)
        sess.run(model.optimizer, feed_dict={model.inputs: examples, model.sameness: sameness})


def predict(examples):
    return sess.run(model.classifier.layers[-1], feed_dict={model.inputs: examples})


def classify(examples):
    predictions = predict(examples)
    return np.argmax(predictions, axis=-1)


def eval():
    examples, sameness = data.batch()
    return sess.run(model.loss, feed_dict={model.inputs: examples, model.sameness: sameness})
