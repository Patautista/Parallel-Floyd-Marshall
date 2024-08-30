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
#define MPI_ROOT 0

void print_matrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int> flatten_matrix(const std::vector<std::vector<int>>& matrix) {
    std::vector<int> flat_matrix;
    for (const auto& row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
    return flat_matrix;
}

void print_vector(std::vector<int> const& input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << "\n";
}

std::vector<int> create_2D_partition(const std::vector<std::vector<int>>& matrix, int& block_size, int& num_blocks) {
    std::vector<int> flat_matrix;
    int n = matrix.size();
    int sqrt_p = static_cast<int>(sqrt(n));
    flat_matrix.resize(n * n);
    for (int p = 0; p < num_blocks; p++) {
        int block_row = (int((p / block_size)));
        int block_col = p % block_size;
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                flat_matrix[p * n + i * block_size + j] = matrix[block_row * block_size + i][block_col * block_size + j];
            }
        }
    }
    return flat_matrix;
}

void write_matrix_to_file(const std::vector<std::vector<int>>& D, int n, int rank, int size, const std::string& filename) {
    int sqrt_p = static_cast<int>(sqrt(size));
    int block_size = n / sqrt_p;
    std::ofstream file;

    file.open(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }


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

int fun(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the input matrix file path is provided as a command-line argument
    if (argc < 2) {
        if (rank == MPI_ROOT) {
            std::cerr << "Usage: " << argv[0] << " <input_matrix_file_path>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file_path = argv[1];

    // Determine the dimensions of the process grid
    int dims[2] = { 0, 0 };
    MPI_Dims_create(size, 2, dims);

    int periods[2] = { 0, 0 };  // No wrap-around in grid
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int n;
    std::vector<std::vector<int>> matrix;

    if (rank == MPI_ROOT) {
        read_matrix_from_file(matrix, input_file_path);
        n = matrix.size();
    }

    // Broadcast the size of the matrix to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block_size = n / dims[0];  // Assuming a square grid and a square matrix

    // Prepare the full matrix for scattering (only needed on rank 0)
    std::vector<int> full_flat_matrix;
    if (rank == MPI_ROOT) {
        full_flat_matrix = create_2D_partition(matrix, block_size, size);
    }

    // Scatter the blocks to all processes
    std::vector<int> local_block(block_size * block_size);
    MPI_Scatter(full_flat_matrix.data(), block_size * block_size, MPI_INT, local_block.data(), block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> local_matrix(block_size, std::vector<int>(block_size));
    // Reconstruct the local matrix block from the flattened data
    for (int i = 0; i < block_size; ++i) {
        std::copy(local_block.begin() + i * block_size, local_block.begin() + (i + 1) * block_size, local_matrix[i].begin());
    }

    // Perform the parallel Floyd-Warshall algorithm
    //floyd_all_pairs_parallel(local_matrix, n, grid_comm);

    int grid_col = rank % block_size;
    int grid_row = int(rank / block_size);
    int sqrt_p = static_cast<int>(sqrt(size));

    // Print local matrix (for debugging)
    std::cout << "Process " << rank << " has submatrix:\n";
    print_matrix(local_matrix);


    // Root process will gather the full matrix
    std::vector<int> full_matrix;
    if (rank == 0) {
        full_matrix.resize(n * n);
    }

    // Calculate the displacements and receive counts for gathering
    std::vector<int> displacements(size, 0);
    std::vector<int> recvcounts(size, block_size * block_size);

    for (int i = 0; i < sqrt_p; ++i) {
        for (int j = 0; j < sqrt_p; ++j) {
            int proc_rank = i * sqrt_p + j;
            displacements[proc_rank] = (i * block_size * n) + (j * block_size);
        }
    }

    // Gather the blocks into the full matrix

    MPI_Gatherv(flatten_matrix(local_matrix).data(), block_size * block_size, MPI_INT,
        full_matrix.data(), recvcounts.data(), displacements.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    // Root process prints the full matrix
    if (rank == 0) {
        std::cout << "\nReconstructed full matrix:\n";
        if (true) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    std::cout << full_matrix[i * n + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    // Construct the output file path
    std::filesystem::path output_file_path = std::filesystem::path(input_file_path).parent_path() / "output_matrix.txt";

    // Write the results to the output file
    write_matrix_to_file(local_matrix, block_size, rank, size, output_file_path.string());

    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
}