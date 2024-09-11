#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <climits>
#include "Matrix.h"

#define INF INT_MAX

class SerialFloydWarshall {
public:
    SerialFloydWarshall(const std::string& input_file_path) : m_input_file_path(input_file_path) {
        // Read the input matrix from the file
        n = calculate_matrix_dimension(m_input_file_path);
        read_matrix_from_file(matrix, input_file_path);
    }

    void execute() {
        floyd_all_pairs();
        print_matrix(matrix);
        write_matrix_to_file(matrix, m_input_file_path + "_result.txt", n);
    }

private:
    int n; // Dimension of the matrix
    std::string m_input_file_path;
    std::vector<std::vector<int>> matrix; // The adjacency matrix

    // The serial Floyd-Warshall algorithm
    void floyd_all_pairs() {
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (matrix[i][k] != INF && matrix[k][j] != INF) {
                        matrix[i][j] = std::min(matrix[i][j], matrix[i][k] + matrix[k][j]);
                    }
                }
            }
        }
    } 
};