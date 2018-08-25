import numpy as np
import matplotlib.pyplot as plt
import data
from manager import predict


def visualize(res=(64, 64)):
    # calculate the box area to plot on
    left = np.min(data.examples[:, 0])
    right = np.max(data.examples[:, 0])
    bottom = np.min(data.examples[:, 1])
    top = np.max(data.examples[:, 1])
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
    for pair in data.must_link:
        pair_examples = data.examples[np.array(pair)]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='k', zorder=1)
    for pair in data.cannot_link:
        pair_examples = data.examples[np.array(pair)]
        plt.plot(pair_examples[:, 0], pair_examples[:, 1], c='w', zorder=1)
    plt.scatter(data.examples[data.classes == 0][:, 0], data.examples[data.classes == 0][:, 1], c='r', zorder=2)
    plt.scatter(data.examples[data.classes == 1][:, 0], data.examples[data.classes == 1][:, 1], c='b', zorder=2)
    plt.show()
