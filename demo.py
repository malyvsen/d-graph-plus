import d_graph_plus


print('Pre-training ARI: ' + str(d_graph_plus.adjusted_rand_index()))
d_graph_plus.train()
print('Training done.')
print('Post-training ARI: ' + str(d_graph_plus.adjusted_rand_index()))
d_graph_plus.visualize.all()
