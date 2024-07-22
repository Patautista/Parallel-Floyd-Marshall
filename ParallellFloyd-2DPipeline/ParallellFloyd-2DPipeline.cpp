#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <fstream>


#define INF INT_MAX

void read_matrix_from_file(std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        int n;
        file >> n;
        matrix.resize(n, std::vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> matrix[i][j];
            }
        }
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
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        read_matrix_from_file(matrix, "C:\\Users\\caleb\\source\\repos\\ParallellFloyd-2DPipeline\\matrix.txt");
        n = matrix.size();
    }

    // Broadcast the size of the matrix to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize local block for each process
    int block_size = n / dims[0];
    int** D = new int* [block_size];
    for (int i = 0; i < block_size; ++i) {
        D[i] = new int[block_size];
    }

    // Distribute the matrix data to all processes
    for (int i = 0; i < block_size; ++i) {
        MPI_Scatter(rank == 0 ? matrix[i].data() : nullptr, block_size, MPI_INT, D[i], block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Perform the parallel Floyd-Warshall algorithm
    floyd_all_pairs_parallel(matrix, matrix.size());

    // ... (rest of the code to gather and print the result)

    for (int i = 0; i < block_size; ++i) {
        delete[] D[i];
    }
    delete[] D;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();
    return 0;
}
