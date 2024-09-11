#pragma once
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

inline void print_vector(std::vector<int> const& input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << "\n";
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
inline int calculate_matrix_dimension(const std::string& file_path) {
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
inline void loop_flat_matrix(const std::vector<int>& matrix, int n, int block_size, int sqrt_p, Func func) {
    for (int block_row = 0; block_row < sqrt_p; ++block_row) {
        for (int row_in_block = 0; row_in_block < block_size; ++row_in_block) {
            for (int block_col = 0; block_col < sqrt_p; ++block_col) {
                for (int col_in_block = 0; col_in_block < block_size; ++col_in_block) {
                    int flat_idx = (block_row * sqrt_p + block_col) * block_size * block_size
                        + row_in_block * block_size + col_in_block;
                    int value = matrix[flat_idx];

                    func(value);
                }
            }
        }
    }
}

inline void print_flat_matrix(const std::vector<int>& matrix, int n, int block_size, int sqrt_p) {
    int counter = 0;
    loop_flat_matrix(matrix, n, block_size, sqrt_p, [&counter, &n](int value) {
        print_element(value); // Print the current element
        counter++; // Increment the counter

        if (counter % n == 0) { // If we've printed 'n' elements, it's time for a newline
            std::cout << std::endl;
        }
        });
    std::cout << std::endl;
}
inline void write_flat_matrix_to_file(const std::vector<int>& matrix, int n, int block_size, int sqrt_p, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    int counter = 0;

    loop_flat_matrix(matrix, n, block_size, sqrt_p, [&counter, &n, &file](int value) {
        write_element_to_file(value, file);
        counter++; // Increment the counter

        if (counter % n == 0) { // If we've printed 'n' elements, it's time for a newline
            file << std::endl;
        }
        });

    file.close();
}
