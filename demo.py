import numpy as np
from manager import train, eval
import visualize
from tasks import current as task


print('Pre-training loss: ' + str(eval()))
train()
print('Training done.')
print('Post-training loss: ' + str(eval()))
visualize.correlation()
