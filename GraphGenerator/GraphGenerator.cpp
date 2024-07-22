#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cstdlib>

void initialize_matrix(std::vector<std::vector<int>>& matrix, int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                matrix[i][j] = rand() % 20 + 1; // random weight between 1 and 20
            }
            else {
                matrix[i][j] = 0; // distance to itself is zero
            }
        }
    }
}

void write_matrix_to_file(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        int n = matrix.size();
        file << n << std::endl;
        for (const auto& row : matrix) {
            for (int value : row) {
                file << value << " ";
            }
            file << std::endl;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

int main() {
    int n = 4; // Example size
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    initialize_matrix(matrix, n);
    write_matrix_to_file(matrix, "../matrix.txt");
    return 0;
}
