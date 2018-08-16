import numpy as np
from manager import train, eval, classify
from viz import visualize
import data


print('Pre-training loss: ' + str(eval()))
print('Training...')
train()
print('Training done.')
print('Post-training loss: ' + str(eval()))
classification = classify(data.examples)
for class_id in range(data.num_classes):
    print('Class ' + str(class_id) + ': ' + str(np.sum(classification == class_id)))
visualize()
