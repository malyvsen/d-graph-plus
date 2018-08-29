import numpy as np
from sklearn.datasets import make_moons


num_examples = 32
num_attributes = 2
num_classes = 2
num_must = 12
num_cannot = 12


examples, classes = make_moons(num_examples)
examples += np.random.normal(scale=5e-2, size=np.shape(examples))

must_link = []
while len(must_link) < num_must:
    a, b = np.random.choice(num_examples, 2, replace=False)
    if classes[a] != classes[b]:
        continue
    if (a, b) in must_link or (b, a) in must_link:
        continue
    must_link.append((a, b))

cannot_link = []
while len(cannot_link) < num_cannot:
    a, b = np.random.choice(num_examples, 2, replace=False)
    if classes[a] == classes[b]:
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
    return 2 / num_classes - 1


def sameness(a, b):
    '''
    a measure of similarity between examples with ids a and b
    takes expert knowledge into account
    '''
    if (a, b) in must_link or (b, a) in must_link:
        return 1
    if (a, b) in cannot_link or (b, a) in cannot_link:
        return -1
    return weight(examples[a], examples[b]) * 1e-2


def batch(size=num_examples):
    included_ids = np.random.choice(num_examples, size=size, replace=False)
    included_examples = examples[included_ids]
    included_sameness = np.array([[sameness(a, b) for b in included_ids] for a in included_ids])
    return included_examples, included_sameness
