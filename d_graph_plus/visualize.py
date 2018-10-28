import numpy as np
import matplotlib.pyplot as plt
from d_graph_plus.data import batch
from d_graph_plus.manager import correlation_matrix
from d_graph_plus.tasks import current as task


def all():
    correlation(show=False)
    if hasattr(task, 'visualize'):
        task.visualize(show=False)
    plt.show()


def correlation(show=True):
    print('Visualizing correlation matrix...')
    correlation = correlation_matrix()
    plt.figure()
    plt.title(f'Correlation matrix (determinant: {np.linalg.det(correlation)})')
    plt.imshow(correlation)
    plt.xlabel('Model-assigned bucket')
    plt.ylabel('Ground-truth class')
    plt.colorbar()
    if show:
        plt.show()
