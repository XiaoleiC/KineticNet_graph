import dgl
from dataloading import METR_LAGraphDataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g = METR_LAGraphDataset()
g0 = g.clone()  # Clone the graph to preserve original
g0 = dgl.remove_self_loop(g0)  # Remove self-loops if any
print(g.in_degrees())
print(g.out_degrees())
print(g0.in_degrees())
print(g0.out_degrees())


# g0 = g.clone()
# g_reverse = dgl.reverse(g, copy_ndata=True, copy_edata=True)
# # g = dgl.remove_self_loop(g)
# g.ndata['out_degree'] = g.out_degrees().float()  # [N]
# g.ndata['in_degree'] = g.in_degrees().float()  # [N]
# g_reverse.ndata['out_degree'] = g_reverse.out_degrees().float()  # [N]
# g_reverse.ndata['in_degree'] = g_reverse.in_degrees().float()  # [N]
# N = g.num_nodes()
# Q = 4

# print(g.has_edges_between(g.nodes(), g.nodes()).any())

# print(g.num_edges())
# print(g_reverse.num_edges())
# print(g.ndata['out_degree'])
# print(g_reverse.ndata['in_degree'])

# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
# ax[0].hist(g.ndata['out_degree'].numpy(), bins=50, color='blue', alpha=0.7)
# ax[1].hist(g_reverse.ndata['in_degree'].numpy(), bins=50, color='red', alpha=0.7)

# plt.savefig('degree_histograms.png')
# plt.close()

# with g.local_scope():
#     # Store distribution on nodes
#     g.ndata['f'] = torch.rand([N,Q])  # [N, Q]
#     reverse_graph = dgl.reverse(g, copy_ndata=True, copy_edata=True)
    
#     # Precompute velocity matrix for broadcasting: [Q] -> [1, Q]
#     xi_broadcast = self.xi_velocities.unsqueeze(0)  # [1, Q]
    
#     # Inflow computation (vectorized over all velocity components)
#     def message_func_inflow(edges):
#         # f_diff: [E, Q] - differences for all velocity components
#         f_diff = edges.dst['f'] - edges.src['f']  # [E, Q]
        
#         # Edge weights and q_ij: [E] -> [E, 1] for broadcasting
#         edge_weight = edges.data.get('weight', torch.ones(edges.src['f'].shape[0], device=f_diff.device))
#         edge_weight = edge_weight.unsqueeze(1)  # [E, 1]
        
#         q_ij = (1.0 / edges.src['out_degree']).unsqueeze(1)  # [E, 1], !!! The problem is that we assume the weight is the same for all particle velocities
        
#         # Broadcast xi_velocities: [1, Q] -> [E, Q]
#         xi_expanded = xi_broadcast.expand(f_diff.shape[0], -1)  # [E, Q]
        
#         # Compute transport for all velocity components: [E, Q]
#         transport = q_ij * xi_expanded * f_diff * edge_weight  # [E, Q]
        
#         return {'transport_all': transport}
    
#     def reduce_func(nodes):
#         # nodes.mailbox['transport_all']: [N, max_degree, Q]
#         # Sum over neighbors (dim=1): [N, Q]
#         return {'transport_sum_all': torch.sum(nodes.mailbox['transport_all'], dim=1)}
    
#     # Single message passing for inflow (all velocity components)
#     g.update_all(message_func_inflow, reduce_func)