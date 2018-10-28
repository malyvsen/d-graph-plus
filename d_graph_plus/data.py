import numpy as np
import random
from tqdm import trange, tqdm
from d_graph_plus.tasks import current as task


print(f'Preparing {task.num_must} ground-truth must-link pairs...')
must_link = []
while len(must_link) < task.num_must:
    a, b = np.random.choice(len(task.examples), 2, replace=False)
    if task.classes[a] != task.classes[b]:
        continue
    if {a, b} in must_link:
        continue
    must_link.append({a, b})

print(f'Preparing {task.num_cannot} ground-truth cannot-link pairs...')
cannot_link = []
while len(cannot_link) < task.num_cannot:
    a, b = np.random.choice(len(task.examples), 2, replace=False)
    if task.classes[a] == task.classes[b]:
        continue
    if {a, b} in cannot_link:
        continue
    cannot_link.append({a, b})


print(f'Preparing neighborhood ranking...')
neighborhood = []
for example in tqdm(task.examples):
    by_distance = sorted(range(len(task.examples)), key=lambda x: np.linalg.norm(example - task.examples[x]))
    neighborhood.append(by_distance[1 : task.neighborhood_size + 1]) # do not include self
# a must only be in neighborhood of b if b is in neighborhood of a
print(f'Guaranteeing mutuality...')
for id in trange(len(neighborhood)):
    neighborhood[id] = list(filter(lambda x: id in neighborhood[x], neighborhood[id]))


def rbf(a, b):
    '''
    measure of similarity between examples with ids a and b
    does not take neighborhood information into account
    radial basis function: https://www.wikiwand.com/en/Radial_basis_function
    '''
    gamma = 1
    return np.exp(-gamma * np.sum(np.square(task.examples[a] - task.examples[b])))


def weight(a, b):
    '''
    a measure of similarity between examples with ids a and b
    takes neighborhood information into account
    equation (5) in original paper
    '''
    if a in neighborhood[b]:
        return 2 * rbf(a, b)
    return 2 / task.num_classes - 1 # if not in neighborhood, assume we know nothing - equal probability over classes


def sameness(a, b):
    '''
    a measure of similarity between examples with ids a and b
    takes expert knowledge into account
    '''
    if {a, b} in must_link:
        return 1
    if {a, b} in cannot_link:
        return -1
    return weight(a, b) * task.auxiliary_weight


def batch():
    included_ids = set()
    for pair in random.sample(must_link, task.min_batch_must):
        included_ids |= pair
    for pair in random.sample(cannot_link, task.min_batch_cannot):
        included_ids |= pair
    while len(included_ids) < task.batch_size:
        included_ids.add(random.randrange(len(task.examples)))
    included_ids = np.array(list(included_ids))
    included_examples = task.examples[included_ids]
    included_sameness = np.array([[sameness(a, b) for b in included_ids] for a in included_ids])
    return included_examples, included_sameness
