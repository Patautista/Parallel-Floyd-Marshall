#pragma once
#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>

#define INF INT_MAX

inline void print_matrix(const std::vector<std::vector<int>>& matrix) {
    std::stringstream stream;
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            stream << elem << " ";
        }
        stream << std::endl;
    }
    std::cout << stream.str();
}

inline std::stringstream print_vector(std::vector<int> const& input)
{
    std::stringstream stream;
    for (int i = 0; i < input.size(); i++) {
        stream << input.at(i) << ' ';
    }
    stream << "\n";
    return stream;
}

inline void read_matrix_from_file(std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for reading." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        int value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

}
// Writes the result matrix to a file
inline void write_matrix_to_file(std::vector<std::vector<int>>& matrix, const std::string& output_file_path, int n) {
    std::ofstream file(output_file_path);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (matrix[i][j] == INF) {
                file << "INF ";
            }
            else {
                file << matrix[i][j] << " ";
            }
        }
        file << std::endl;
    }
}

// Reads the dimension of the matrix from the file
inline int calculate_matrix_dimension_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        int count = 0;
        int value;
        while (iss >> value) {
            count++;
        }
        return count;
    }

    std::cerr << "Error reading file." << std::endl;
    exit(1);
    return -1;
}
inline void print_element(int value) {
    if (value == INF) {
        std::cout << "INF ";
    }
    else {
        std::cout << value << " ";
    }
}

inline void write_element_to_file(int value, std::ofstream& file) {
    if (value == INF) {
        file << "INF ";
    }
    else {
        file << value << " ";
    }
}

template <typename Func>
inline void loop_flat_matrix(const std::vector<int>& matrix, int n, const std::vector<int>& row_block_sizes, const std::vector<int>& col_block_sizes, int sqrt_p, Func func) {
    // Total number of elements
    int total_elements = n * n;

    // Calculate block offsets (prefix sums of block sizes)
    std::vector<int> row_offsets(sqrt_p + 1, 0);
    std::vector<int> col_offsets(sqrt_p + 1, 0);

    for (int i = 1; i <= sqrt_p; ++i) {
        row_offsets[i] = row_offsets[i - 1] + row_block_sizes[i - 1];
        col_offsets[i] = col_offsets[i - 1] + col_block_sizes[i - 1];
    }

    #pragma omp parallel for
    for (int global_index = 0; global_index < total_elements; ++global_index) {
        // Calculate global row and column
        int global_row = global_index / n;
        int global_col = global_index % n;

        // Find which block the global row and column belong to
        int block_row = 0, block_col = 0;
        for (int i = 0; i < sqrt_p; ++i) {
            if (global_row >= row_offsets[i] && global_row < row_offsets[i + 1]) {
                block_row = i;
            }
            if (global_col >= col_offsets[i] && global_col < col_offsets[i + 1]) {
                block_col = i;
            }
        }

        // Local row and column within the block
        int row_in_block = global_row - row_offsets[block_row];
        int col_in_block = global_col - col_offsets[block_col];

        // Calculate the block index and the index within the block
        int block_index = block_row * sqrt_p + block_col;
        int index_in_block = row_in_block * col_block_sizes[block_col] + col_in_block;

        // Calculate the final flat index for the flattened matrix
        int flat_index = 0;
        for (int i = 0; i < block_index; ++i) {
            flat_index += row_block_sizes[i / sqrt_p] * col_block_sizes[i % sqrt_p];
        }
        flat_index += index_in_block;

        // Call the provided function with the value and its global position
        func(matrix[flat_index]);
    }
}



inline void print_flat_matrix(const std::vector<int>& matrix, int n, const std::vector<int>& row_block_sizes, const std::vector<int>& col_block_sizes, int sqrt_p) {
    int counter = 0;
    loop_flat_matrix(matrix, n, row_block_sizes, col_block_sizes, sqrt_p, [&counter, &n](int value) {
        print_element(value); // Print the current element
        counter++; // Increment the counter

        if (counter % n == 0) { // If we've printed 'n' elements, it's time for a newline
            std::cout << std::endl;
        }
        });
    std::cout << std::endl;
}
inline void write_flat_matrix_to_file(const std::vector<int>& matrix, int n, const std::vector<int>& row_block_sizes, const std::vector<int>& col_block_sizes, int sqrt_p, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    file.close();
}