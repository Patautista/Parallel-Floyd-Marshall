#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <ctime>

namespace fs = std::filesystem;

class GraphGenerator {
public:
    // Constructor to initialize matrix size
    GraphGenerator(int n) : n(n), matrix(n, std::vector<int>(n)) {}

    // Initialize matrix with random weights
    void initialize_matrix() {
        srand(time(0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    matrix[i][j] = rand() % 20 + 1;  // random weight between 1 and 20
                }
                else {
                    matrix[i][j] = 0;  // distance to itself is zero
                }
            }
        }
    }

    // Write matrix to file
    void write_matrix_to_file(const std::string& base_path = "samples") {
        // Ensure the base samples directory exists
        fs::path samples_dir(base_path);
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

        // Write the matrix to a file
        std::string filename = (new_folder_path / "matrix.txt").string();
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& row : matrix) {
                for (int value : row) {
                    file << value << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "Matrix written to " << filename << std::endl;
        }
        else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }

private:
    int n;  // Matrix size
    std::vector<std::vector<int>> matrix;  // The matrix to store weights
};
