import numpy as np
import matplotlib.pyplot as plt
from d_graph_plus.data import batch
from d_graph_plus.manager import predict
from d_graph_plus.tasks import current as task


def correlation():
    print('Visualizing correlation matrix...')
    output = predict(task.examples)
    correlation = []
    for class_id in range(task.num_classes):
        class_correlation = np.mean(output[task.classes == class_id], axis=0)
        correlation.append(class_correlation)

    plt.title('Correlation matrix')
    plt.imshow(np.array(correlation))
    plt.xlabel('Model-assigned bucket')
    plt.ylabel('Ground-truth class')
    plt.colorbar()
    plt.show()
