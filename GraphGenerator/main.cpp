#include <iostream>
#include "GraphGenerator.cpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);  // Matrix size from command-line argument

    // Create a GraphGenerator object and generate the matrix
    GraphGenerator graphGenerator(n);
    graphGenerator.initialize_matrix();

    // Write the matrix to a file
    graphGenerator.write_matrix_to_file();

    return 0;
}