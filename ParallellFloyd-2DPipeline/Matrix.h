#pragma once
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>

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
