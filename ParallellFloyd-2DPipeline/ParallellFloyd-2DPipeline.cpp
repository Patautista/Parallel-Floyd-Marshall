#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>

#define INF INT_MAX

void write_matrix_to_file(const std::vector<std::vector<int>>& D, int n, int rank, int size, const std::string& filename) {
    int block_size = n / size;
    std::ofstream file;

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return;
        }
    }

    // Temporary buffer for gathering the matrix blocks at root process
    int* temp_matrix = new int[n * n];
    for (int i = 0; i < block_size; ++i) {
        MPI_Gather(D[i].data(), block_size, MPI_INT, rank == 0 ? &temp_matrix[i * n] : nullptr, block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Write the matrix to the file
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file << temp_matrix[i * n + j] << " ";
            }
            file << std::endl;
        }
        file.close();
    }

    delete[] temp_matrix;
}

void read_matrix_from_file(std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        std::vector<std::vector<int>> temp_matrix;

        // Read the file line by line
        while (std::getline(file, line)) {
            std::vector<int> row;
            std::stringstream ss(line);
            int value;
            while (ss >> value) {
                row.push_back(value);
            }
            temp_matrix.push_back(row);
        }

        // Check if the matrix is square
        size_t n = temp_matrix.size();
        for (const auto& row : temp_matrix) {
            if (row.size() != n) {
                std::cerr << "Error: Matrix is not square." << std::endl;
                return;
            }
        }

        // Assign the temporary matrix to the output matrix
        matrix = temp_matrix;

        file.close();
    }
    else {
        std::cerr << "Unable to open file for reading." << std::endl;
    }
}

void floyd_all_pairs_parallel(std::vector<std::vector<int>>& D, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrt_p = static_cast<int>(sqrt(size));
    int row_per_proc = n / sqrt_p;

    std::vector<int> row_buffer(n);
    std::vector<int> col_buffer(row_per_proc);

    for (int k = 0; k < n; ++k) {
        int owner_row = k / row_per_proc;
        if (rank == owner_row) {
            for (int j = 0; j < n; ++j) {
                row_buffer[j] = D[k % row_per_proc][j];
            }
        }
        MPI_Bcast(row_buffer.data(), n, MPI_INT, owner_row, MPI_COMM_WORLD);

        int owner_col = k / row_per_proc;
        if (rank % sqrt_p == owner_col) {
            for (int i = 0; i < row_per_proc; ++i) {
                col_buffer[i] = D[i][k];
            }
        }
        MPI_Bcast(col_buffer.data(), row_per_proc, MPI_INT, owner_col, MPI_COMM_WORLD);

        for (int i = 0; i < row_per_proc; ++i) {
            for (int j = 0; j < n; ++j) {
                if (D[i][j] > col_buffer[i] + row_buffer[j]) {
                    D[i][j] = col_buffer[i] + row_buffer[j];
                }
            }
        }

        // Print debug information
        if (rank == 0) {
            std::cout << "Iteration k=" << k << "\n";
            std::cout << "Row buffer: ";
            for (const auto& val : row_buffer) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes sync before printing column buffer

        if (rank == 0) {
            std::cout << "Column buffer from process " << owner_col << ": ";
            for (const auto& val : col_buffer) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes sync before next iteration
    }

    // Print final matrix
    if (rank == 0) {
        std::cout << "Final distance matrix:\n";
        for (const auto& row : D) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the input matrix file path is provided as a command-line argument
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_matrix_file_path>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file_path = argv[1];

    // Assuming we are using a square process grid
    int dims[2] = { 0, 0 };
    MPI_Dims_create(size, 2, dims);

    int periods[2] = { 0, 0 };
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);

    int n;
    std::vector<std::vector<int>> matrix;

    // Only the root process reads the matrix from the file
    if (rank == 0) {
        read_matrix_from_file(matrix, input_file_path);
        n = matrix.size();
    }

    // Broadcast the size of the matrix to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize local block for each process
    int block_size = n / dims[0];
    int** matrix_buffer = new int* [block_size];
    for (int i = 0; i < block_size; ++i) {
        matrix_buffer[i] = new int[block_size];
    }

    // Distribute the matrix data to all processes
    for (int i = 0; i < block_size; ++i) {
        MPI_Scatter(rank == 0 ? matrix[i].data() : nullptr, block_size, MPI_INT, matrix_buffer[i], block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Perform the parallel Floyd-Warshall algorithm
    floyd_all_pairs_parallel(matrix, matrix.size());

    // Create the output file path in the same directory as the input file
    std::filesystem::path output_file_path = std::filesystem::path(input_file_path).parent_path() / "output_matrix.txt";

    // Write the results to the output file
    write_matrix_to_file(matrix, n, rank, size, output_file_path.string());

    for (int i = 0; i < block_size; ++i) {
        delete[] matrix_buffer[i];
    }
    delete[] matrix_buffer;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();
    return 0;
}
