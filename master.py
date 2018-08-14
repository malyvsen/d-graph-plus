from train import train, eval
# from viz import visualize


print('Pre-training loss: ' + str(eval()))
print('Training...')
train()
print('Training done.')
print('Post-training loss: ' + str(eval()))
# print('Visualizing result...')
# visualize()
