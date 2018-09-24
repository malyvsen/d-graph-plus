import numpy as np
from d_graph_plus.tasks import current as task


print(f'Preparing {task.num_must} must-link pairs...')
must_link = []
while len(must_link) < task.num_must:
    a, b = np.random.choice(len(task.examples), 2, replace=False)
    if task.classes[a] != task.classes[b]:
        continue
    if (a, b) in must_link or (b, a) in must_link:
        continue
    must_link.append((a, b))

print(f'Preparing {task.num_cannot} cannot-link pairs...')
cannot_link = []
while len(cannot_link) < task.num_cannot:
    a, b = np.random.choice(len(task.examples), 2, replace=False)
    if task.classes[a] == task.classes[b]:
        continue
    if (a, b) in cannot_link or (b, a) in cannot_link:
        continue
    cannot_link.append((a, b))


def rbf(a, b):
    '''
    radial basis function: https://www.wikiwand.com/en/Radial_basis_function
    a measure of similarity between examples a and b
    '''
    gamma = 1
    return np.exp(-gamma * np.sum(np.square(a - b)))


def weight(a, b):
    '''
    see equation (5) in original paper
    a measure of similarity between examples a and b
    '''
    epsilon = 0.5
    rbf_val = rbf(a, b)
    if rbf_val > epsilon:
        return 2 * rbf_val - 1
    return 2 / task.num_classes - 1


def sameness(a, b):
    '''
    a measure of similarity between examples with ids a and b
    takes expert knowledge into account
    '''
    if (a, b) in must_link or (b, a) in must_link:
        return 1
    if (a, b) in cannot_link or (b, a) in cannot_link:
        return -1
    return weight(task.examples[a], task.examples[b]) * 1e-2


def batch(size=task.batch_size):
    included_ids = np.random.choice(len(task.examples), size=size, replace=False)
    included_examples = task.examples[included_ids]
    included_sameness = np.array([[sameness(a, b) for b in included_ids] for a in included_ids])
    return included_examples, included_sameness
