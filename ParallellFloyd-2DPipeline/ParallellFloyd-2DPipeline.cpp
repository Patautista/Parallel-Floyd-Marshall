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
    for (int k = 0; k < num_blocks; k++) {
        int block_row = (int((k / block_size)));
        int block_col = k % block_size;
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                flat_matrix[k * n + i * block_size + j] = matrix[block_row * block_size + i][block_col * block_size + j];
            }
        }
    }
    return flat_matrix;
}

void write_matrix_to_file(const std::vector<std::vector<int>>& D, int n, int rank, int size, const std::string& filename) {
    int sqrt_p = static_cast<int>(sqrt(size));
    int block_size = n / sqrt_p;
    std::ofstream file;

    if (rank == 0) {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return;
        }
    }

    // Temporary buffer for gathering the matrix blocks at the root process
    std::vector<int> temp_matrix(rank == 0 ? n * n : 0);

    // Gather the blocks from all processes
    for (int i = 0; i < block_size; ++i) {
        MPI_Gather(D[i].data(), block_size, MPI_INT,
            rank == 0 ? &temp_matrix[i * n] : nullptr,
            block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Reconstruct and write the full matrix to the file
        for (int i = 0; i < sqrt_p; ++i) {
            for (int j = 0; j < sqrt_p; ++j) {
                int block_start_row = i * block_size;
                int block_start_col = j * block_size;

                for (int row = 0; row < block_size; ++row) {
                    for (int col = 0; col < block_size; ++col) {
                        int global_row = block_start_row + row;
                        int global_col = block_start_col + col;
                        file << temp_matrix[global_row * n + global_col] << " ";
                    }
                    file << std::endl;
                }
            }
        }
        file.close();
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

void floyd_all_pairs_parallel(std::vector<std::vector<int>>& local_matrix, int n, MPI_Comm& comm) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrt_p = static_cast<int>(sqrt(size));
    int block_size = n / sqrt_p;
   
    std::vector<int> global_row_buffer(n);
    std::vector<int> global_col_buffer(n);

    int grid_col = rank % sqrt_p;
    int grid_row = int(rank / block_size);

    //std::cout << rank << ": Floyd start" << "(" << n << ")\n";

    // This loop iterates through each vertex k, 
    // treating it as an intermediate vertex in potential shortest paths between all pairs of vertices.
    for (int k = 1; k < n; ++k) {   

        // Row Responsibility: owner_row determines which process is responsible for broadcasting a particular row k. 
        // If the current process is responsible, it fills row_buffer with that row.

        // adds extra range for last row
        int extra_range = (grid_row == sqrt_p - 1 ? 1 : 0);
        int k_grid_row = int(k / block_size);
        int last_row_owner = (k_grid_row * block_size) + block_size - 1;
        if (int(rank / sqrt_p) < k && k < int(rank / sqrt_p) + block_size + extra_range) {
            int local_row_index = k % block_size;
            for (int i = 0; i < block_size; i++) {
                global_row_buffer[(grid_col * block_size) + i] = local_matrix[local_row_index][i];
            }

            if (grid_col > 0) {
                int coords[2];
                MPI_Cart_coords(comm, rank, 2, coords);
                coords[1]--;
                int rec_partner;
                MPI_Cart_rank(comm, coords, &rec_partner);

                std::vector<int> temp(block_size * grid_col);
                std::cout << "\niteration " << k << " : " << rank << " receives row from " << rec_partner << "\n";
                MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //print_vector(temp);
                std::cout << "----------------\n\n";
                std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
            }

            int coords[2];
            MPI_Cart_coords(comm, rank, 2, coords);
            coords[1]++;
            //std::cout << "partner coords! " << coords[0] << " " << coords[1];
            if (coords[1] < block_size) {
                int partner;
                MPI_Cart_rank(comm, coords, &partner);
                std::cout << "\niteration " << k << " : " << rank << " sends row to " << partner << "\n";
                //print_vector(global_row_buffer);
                std::cout << "----------------\n\n";
                MPI_Send(global_row_buffer.data(), block_size * (grid_col + 1), MPI_INT, partner, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Bcast(global_row_buffer.data(), n, MPI_INT, last_row_owner, MPI_COMM_WORLD);
        
        // Column Responsibility: owner_col determines which process is responsible for broadcasting a particular column k. 
        // If the current process is responsible, it fills col_buffer with that column.
        // rank % sqrt_p < k < (rank % sqrt_p + block_size)

        // adds extra range for last column
        extra_range = (grid_col == sqrt_p - 1 ? 1 : 0);
        // sets column broadcaster index
        int last_col_owner = size - block_size + int(k / block_size);
        // is column owner
        if (((rank % sqrt_p) < k && (k < rank % sqrt_p + block_size) + extra_range)) {

            if (grid_row > 0) {
                int rec_partner = rank - block_size;
                std::vector<int> temp(block_size * grid_row);
                std::cout << "\niteration " << k << " : " << rank << " receives column from " << rec_partner << "\n";
                MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //print_vector(temp);
                std::cout << "----------------\n\n";
                std::copy(temp.begin(), temp.end(), global_col_buffer.begin());
            }

            int local_col_index = k % block_size;
            for (int i = 0; i < block_size; i++) {
                global_col_buffer[(grid_row * block_size) + i] = local_matrix[i][local_col_index];
            }
            int partner = rank + block_size;
            if (partner < size) {
                std::cout << "\niteration " << k << " : " << rank << " sends column to " << partner << "\n";
                //print_vector(global_col_buffer);
                std::cout << "----------------\n\n";
                MPI_Send(global_col_buffer.data(), block_size * (grid_row + 1), MPI_INT, partner, 0, MPI_COMM_WORLD);
            }
        }
        MPI_Bcast(global_col_buffer.data(), n, MPI_INT, last_col_owner, MPI_COMM_WORLD);

        // Each process updates its block of the matrix D by comparing the current distance D[i][j]
        // with the potential shorter path col_buffer[i] + row_buffer[j]. 
        // If the new path is shorter, it updates D[i][j].


        //...
        
        if (rank == 0) {
            std::cout << "rank " << rank << " view:" << "\n";
            std::cout << "iteration " << k << ":\n";
            //std::cout << "last_col_owner " << last_col_owner << ":\n";
            print_vector(global_row_buffer);
            print_vector(global_col_buffer);
            std::cout << "\n\n";
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

    if (rank == MPI_ROOT) {
        //std::cout << "block_size: " << block_size << "\n";
    }

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

    /*
        std::cout << "--" << rank << "-- START \n";
        print_matrix(local_matrix);
        std::cout << "\n--" << rank << "-- END \n";
    */

    // Perform the parallel Floyd-Warshall algorithm
    floyd_all_pairs_parallel(local_matrix, n, grid_comm);

    // Construct the output file path
    std::filesystem::path output_file_path = std::filesystem::path(input_file_path).parent_path() / "output_matrix.txt";

    // Write the results to the output file
    //write_matrix_to_file(local_matrix, block_size, rank, size, output_file_path.string());

    // Gather all blocks to the root process
    std::vector<int> global_matrix(rank == 0 ? n * n : 0);

    for (int i = 0; i < block_size; ++i) {
        //MPI_Gather(local_matrix[i].data(), block_size, MPI_INT, rank == 0 ? &global_matrix[i * n] : nullptr, n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Print the full matrix from the root process
    if (rank == 0) {
        //std::cout << "Full matrix view:\n";
        //print_vector(global_matrix);
    }

    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}
