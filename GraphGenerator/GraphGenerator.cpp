#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sstream>

namespace fs = std::filesystem;

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]); // Matrix size from command-line argument
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    initialize_matrix(matrix, n);

    std::string base_path = "samples";
    fs::path samples_dir(base_path);

    // Ensure the base samples directory exists
    if (!fs::exists(samples_dir)) {
        fs::create_directories(samples_dir);
    }

    // Find the last folder number and create a new folder with the next number
    int max_folder_number = 0;
    for (const auto& entry : fs::directory_iterator(samples_dir)) {
        if (entry.is_directory()) {
            int folder_number = std::stoi(entry.path().filename().string());
            if (folder_number > max_folder_number) {
                max_folder_number = folder_number;
            }
        }
    }

    int new_folder_number = max_folder_number + 1;
    fs::path new_folder_path = samples_dir / std::to_string(new_folder_number);
    fs::create_directories(new_folder_path);

    std::string filename = (new_folder_path / "matrix.txt").string();
    write_matrix_to_file(matrix, filename);

    std::cout << "Matrix written to " << filename << std::endl;

    return 0;
}
