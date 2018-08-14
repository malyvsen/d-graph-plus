import numpy as np
from sklearn.datasets import make_moons


num_examples = 100
num_attributes = 2
num_classes = 2
num_constraints = 20


examples, classes = make_moons(num_examples)

sameness = np.identity(num_examples) # must link each point with itself
for i in range(num_constraints):
    endpoints = np.random.choice(num_examples, size=2, replace=False)
    if classes[endpoints[0]] == classes[endpoints[0]]:
        # must link
        sameness[endpoints[0], endpoints[1]] = 1
        sameness[endpoints[1], endpoints[0]] = 1
    else:
        # cannot link
        sameness[endpoints[0], endpoints[1]] = -1
        sameness[endpoints[1], endpoints[0]] = -1


def batch(size=num_examples):
    included_ids = np.random.choice(num_examples, size=size, replace=False)
    included_examples = examples[included_ids]
    included_sameness = sameness[included_ids][:, included_ids]
    return included_examples, included_sameness
