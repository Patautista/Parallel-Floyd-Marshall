import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os

def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Determine the size of the matrix
    n = len(lines)
    
    # Initialize an empty adjacency matrix
    matrix = np.zeros((n, n), dtype=int)
    
    # Fill the adjacency matrix
    for i, line in enumerate(lines):
        row = list(map(int, line.split()))
        for j, value in enumerate(row):
            matrix[i, j] = value
    
    return matrix

def create_graph_from_matrix(matrix):
    G = nx.DiGraph()
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Limit to upper triangular part
            if matrix[i, j] != 0 and matrix[i, j] != np.inf:  # Skip zero weights and infinities
                G.add_edge(i, j, weight=matrix[i, j])
    return G

def visualize_graph(G, output_path, filename):
    pos = nx.circular_layout(G)
    edges = G.edges(data=True)

    plt.figure(figsize=(24, 16))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})
    plt.title(f'Graph Visualization of {filename}')
    plt.savefig(os.path.join(output_path, filename))  # Save as PNG
    plt.close()

def save_matrix_to_file(matrix, filename):
    np.savetxt(filename, matrix, fmt='%d')

def main():
    parser = argparse.ArgumentParser(description='Visualize shortest path distance matrix.')
    parser.add_argument('subfolder', type=str, help='Subfolder name for input matrices and output files')
    args = parser.parse_args()

    # Set the path to the subfolder
    subfolder_path = os.path.join(os.getcwd(), f"samples\\{args.subfolder}")
    
    # Ensure the subfolder exists
    if not os.path.exists(subfolder_path):
        print(f"Subfolder '{args.subfolder}' does not exist.")
        return

    # Construct matrix file path and read matrix
    matrix_file_path = os.path.join(subfolder_path, 'matrix.txt')
    if not os.path.exists(matrix_file_path):
        print(f"Matrix file '{matrix_file_path}' does not exist.")
        return

    matrix = read_matrix_from_file(matrix_file_path)
    
    # Save matrix to the subfolder (although it's already present)
    save_matrix_to_file(matrix, matrix_file_path)
    
    # Create graph from matrix
    G = create_graph_from_matrix(matrix)
    
    # Save the input matrix graph visualization
    visualize_graph(G, subfolder_path, 'input_graph.png')

    # Save the output matrix
    output_matrix_file_path = os.path.join(subfolder_path, 'output_matrix.txt')
    save_matrix_to_file(matrix, output_matrix_file_path)
    
    # Create and save graph from the output matrix
    output_matrix = read_matrix_from_file(output_matrix_file_path)
    G_output = create_graph_from_matrix(output_matrix)
    visualize_graph(G_output, subfolder_path, 'output_graph.png')

if __name__ == '__main__':
    main()
