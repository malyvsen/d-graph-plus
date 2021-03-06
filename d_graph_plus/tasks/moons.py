import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons


# dataset
num_attributes = 2
num_classes = 2

examples, classes = make_moons(256)
examples += np.random.normal(scale=5e-2, size=np.shape(examples))


# hyperparameters
num_must = 12
num_cannot = 12
neighborhood_size = 64
auxiliary_weight = 1e-2
gamma = 1 # parameter for RBF
regularizer = tf.contrib.layers.l2_regularizer(1e-2)
batch_size = len(examples) # examples
min_batch_must = 0 # pairs of examples
min_batch_cannot = 0 # pairs of examples
learning_rate = 1e-4
num_episodes = 4096


class model:
    input = tf.placeholder(tf.float32, shape=(None, num_attributes))
    layers = [input]
    layers.append(tf.layers.dense(inputs=layers[-1], units=16, activation=tf.nn.relu, kernel_regularizer=regularizer))
    layers.append(tf.layers.dense(inputs=layers[-1], units=4, activation=tf.nn.relu, kernel_regularizer=regularizer))
    output = tf.layers.dense(inputs=layers[-1], units=num_classes, activation=tf.nn.softmax, kernel_regularizer=regularizer)
    layers.append(output)


def visualize(show=True, res=(64, 64)):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('classification space')
    from d_graph_plus.manager import predict
    from d_graph_plus.data import must_link, cannot_link

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
        pair_examples = examples[np.array(list(pair))]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='k', zorder=1)
    for pair in cannot_link:
        pair_examples = examples[np.array(list(pair))]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='w', zorder=1)
    plt.scatter(examples[classes == 0][:, 0], examples[classes == 0][:, 1], c='r', zorder=2)
    plt.scatter(examples[classes == 1][:, 0], examples[classes == 1][:, 1], c='b', zorder=2)
    if show:
        plt.show()
