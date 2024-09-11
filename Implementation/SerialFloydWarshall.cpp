#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <climits>
#include "Matrix.h"
#include "../FloydWarshallAllPairs/FloydOptions.cpp"

#define INF INT_MAX

class SerialFloydWarshall {
public:
    SerialFloydWarshall(FloydOptions& options) : m_options(options){
        // Read the input matrix from the file
        n = calculate_matrix_dimension(m_options.InputPath);
        read_matrix_from_file(matrix, m_options.InputPath);
    }

    void execute() {
        floyd_all_pairs();
        print_matrix(matrix);
        write_matrix_to_file(matrix, m_options.InputPath + "_result.txt", n);
    }

private:
    int n; // Dimension of the matrix
    FloydOptions m_options;
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