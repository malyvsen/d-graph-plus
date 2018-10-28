import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


# dataset
num_attributes = 2
num_classes = 2

examples, classes = make_moons(32)
examples += np.random.normal(scale=5e-2, size=np.shape(examples))


# hyperparameters
num_must = 12
num_cannot = 12
neighborhood_size = 8
auxiliary_weight = 1e-2
regularization = 0
batch_size = len(examples) # examples
min_batch_must = 0 # pairs of examples
min_batch_cannot = 0 # pairs of examples
learning_rate = 1e-4
num_episodes = 4096


class model:
    input = tf.placeholder(tf.float32, shape=(None, num_attributes))
    layers = [input]
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    layers.append(tf.layers.dense(inputs=layers[-1], units=8, activation=tf.nn.relu))
    output = tf.layers.dense(inputs=layers[-1], units=num_classes, activation=tf.nn.softmax)
    layers.append(output)
    l2_penalty = None # TODO: replace placeholder with actual value like for mnist


def visualize(res=(64, 64)):
    from d_graph_plus.manager import predict
    # calculate the box area to plot on
    left = np.min(examples[:, 0])
    right = np.max(examples[:, 0])
    bottom = np.min(examples[:, 1])
    top = np.max(examples[:, 1])
    half_pixel_x = (right - left) / res[1] / 2
    half_pixel_y = (top - bottom) / res[0] / 2
    extent = (left - half_pixel_x, right + half_pixel_x, bottom - half_pixel_y, top + half_pixel_y)

    # create background
    grid_x, grid_y = np.meshgrid(np.linspace(top, bottom, res[0]), np.linspace(left, right, res[1]))
    grid = np.array([grid_x.flatten(), grid_y.flatten()]).T
    predictions = predict(grid)
    bg = np.reshape([(pred[0], 1.0, pred[1]) for pred in predictions], (res[0], res[1], 3))

    # plot and show
    plt.imshow(bg, zorder=0, interpolation='gaussian', extent=extent)
    for pair in must_link:
        pair_examples = examples[np.array(pair)]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='k', zorder=1)
    for pair in cannot_link:
        pair_examples = examples[np.array(pair)]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='w', zorder=1)
    plt.scatter(examples[classes == 0][:, 0], examples[classes == 0][:, 1], c='r', zorder=2)
    plt.scatter(examples[classes == 1][:, 0], examples[classes == 1][:, 1], c='b', zorder=2)
    plt.show()
