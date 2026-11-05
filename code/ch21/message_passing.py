# One round of graph convolution, by hand, on a 4-node graph.
import torch

#   0 --- 1
#   |   /
#   |  /
#   2 --- 3
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
N = 4
A = torch.zeros(N, N)
for i, j in edges:
    A[i, j] = A[j, i] = 1.0
A_hat = A + torch.eye(N)                                  # every node also keeps its own message
deg = A_hat.sum(1)
D_inv_sqrt = torch.diag(deg ** -0.5)
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt                  # symmetric normalization (Kipf & Welling)

X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])   # 2 features per node
print("degrees (with self-loops):", deg.tolist())
print("normalized adjacency:\n", A_norm.numpy().round(3))
H = A_norm @ X                                            # aggregate neighbours' features
print("after one propagation step:\n", H.numpy().round(3))
print("node 3 (only neighbour: node 2) now holds", H[3].numpy().round(3), "- it 'heard' from node 2")
