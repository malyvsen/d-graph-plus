import d_graph_plus


print('Pre-training objective: ' + str(d_graph_plus.eval()))
d_graph_plus.train()
print('Training done.')
print('Post-training objective: ' + str(d_graph_plus.eval()))
d_graph_plus.visualize.all()
