import numpy as np
import matplotlib.pyplot as plt
from d_graph_plus.data import batch
import d_graph_plus.manager as manager
from d_graph_plus.tasks import current as task


def all():
    correlation(show=False)
    history(show=False)
    if hasattr(task, 'visualize'):
        task.visualize(show=False)
    plt.show()


def correlation(show=True):
    correlation = manager.correlation_matrix()
    plt.figure().canvas.set_window_title('correlation matrix')
    plt.title(f'Correlation matrix (determinant: {np.linalg.det(correlation)})')
    plt.imshow(correlation)
    plt.xlabel('Model-assigned bucket')
    plt.ylabel('Ground-truth class')
    plt.colorbar()
    if show:
        plt.show()


def history(show=True):
    plt.figure().canvas.set_window_title('history')
    plt.title('history')
    for history_of in sorted(manager.history):
        plt.plot(manager.history[history_of], label=history_of)
    plt.legend()
    if show:
        plt.show()
