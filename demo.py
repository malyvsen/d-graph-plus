import d_graph_plus


print('Pre-training loss: ' + str(d_graph_plus.eval()))
d_graph_plus.train()
print('Training done.')
print('Post-training loss: ' + str(d_graph_plus.eval()))
d_graph_plus.visualize.correlation()
