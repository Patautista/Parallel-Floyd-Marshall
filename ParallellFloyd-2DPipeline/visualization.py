import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Read the matrix from the file
matrix = np.loadtxt('output_matrix.txt')

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges with weights
n = matrix.shape[0]
for i in range(n):
    for j in range(n):
        if matrix[i, j] != 0 and matrix[i, j] != np.inf:  # Skip zero weights and infinities
            G.add_edge(i, j, weight=matrix[i, j])

# Draw the graph
pos = nx.spring_layout(G)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})
plt.title('Graph Visualization of Shortest Paths')
plt.show()
