#include "ParallellFloydWarshall.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "Logger.h"

#define INF INT_MAX
#define MPI_ROOT 0

class ParallelFloydWarshall {
public:
    ParallelFloydWarshall(int argc, char** argv) {
        m_logger.enableFileLogging("app.log");
        m_logger.setLogLevel(LogLevel::DEBUG);
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (argc < 2) {
            if (rank == MPI_ROOT) {
                std::cerr << "Usage: " << argv[0] << " <input_matrix_file_path>" << std::endl;
            }
            MPI_Finalize();
            exit(1);
        }

        input_file_path = argv[1];
    }

    ~ParallelFloydWarshall() {
        MPI_Finalize();
    }

private:
    int m_block_size;
    Logger& m_logger = Logger::getInstance();
    std::stringstream m_log_stream;
    void floyd_all_pairs_parallel(std::vector<std::vector<int>>& local_matrix, int n, MPI_Comm& comm) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int sqrt_p = static_cast<int>(sqrt(size));

        std::vector<int> global_row_buffer(n);
        std::vector<int> global_col_buffer(n);

        int grid_col = rank % sqrt_p;
        int grid_row = int(rank / m_block_size);

        // This loop iterates through each vertex k, 
        // treating it as an intermediate vertex in potential shortest paths between all pairs of vertices.
        for (int k = 1; k < n; k++) {

            // Row Responsibility: owner_row determines which process is responsible for broadcasting a particular row k. 
            // If the current process is responsible, it fills row_buffer with that row.

            int k_grid_row = int(k / m_block_size);
            int last_row_owner = (k_grid_row * sqrt_p) + sqrt_p - 1;

            if (should_send_row(k, sqrt_p, grid_row)) {
                int local_row_index = k % m_block_size;

                for (int i = 0; i < m_block_size; i++) {
                    global_row_buffer[(grid_col * m_block_size) + i] = local_matrix[local_row_index][i];
                }

                if (grid_col > 0) {
                    int coords[2];
                    MPI_Cart_coords(comm, rank, 2, coords);
                    coords[1]--;
                    int rec_partner;
                    MPI_Cart_rank(comm, coords, &rec_partner);

                    std::vector<int> temp(m_block_size * grid_col);

                    m_log_stream << "\niteration " << k << " : " << rank << " receives row from " << rec_partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();

                    MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
                }

                int coords[2];
                MPI_Cart_coords(comm, rank, 2, coords);
                coords[1]++;
                if (grid_col + 1 < sqrt_p) {
                    int partner;
                    MPI_Cart_rank(comm, coords, &partner);
                    m_log_stream << "\niteration " << k << " : " << rank << " sends row to " << partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    //print_vector(global_row_buffer);
                    //std::cout << "----------------\n\n";
                    MPI_Send(global_row_buffer.data(), m_block_size * (grid_col + 1), MPI_INT, partner, 0, MPI_COMM_WORLD);
                }
            }

            MPI_Bcast(global_row_buffer.data(), n, MPI_INT, last_row_owner, MPI_COMM_WORLD);

            // Column Responsibility: owner_col determines which process is responsible for broadcasting a particular column k. 
            // If the current process is responsible, it fills col_buffer with that column.

            // adds extra range for last column
            
            // sets column broadcaster index
            int last_col_owner = size - sqrt_p + int(k / m_block_size);

            if (should_send_column(k, sqrt_p, grid_col, rank)) {

                if (grid_row > 0) {
                    int rec_partner = rank - m_block_size;
                    std::vector<int> temp(m_block_size * grid_row);
                    m_log_stream << "\niteration " << k << " : " << rank << " receives column from " << rec_partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //print_vector(temp);
                    //std::cout << "----------------\n\n";
                    std::copy(temp.begin(), temp.end(), global_col_buffer.begin());
                }

                int local_col_index = k % m_block_size;
                for (int i = 0; i < m_block_size; i++) {
                    global_col_buffer[(grid_row * m_block_size) + i] = local_matrix[i][local_col_index];
                }
                int partner = rank + m_block_size;
                if (partner < size) {
                    m_log_stream << "\niteration " << k << " : " << rank << " sends column to " << partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    MPI_Send(global_col_buffer.data(), m_block_size * (grid_row + 1), MPI_INT, partner, 0, MPI_COMM_WORLD);
                }
            }
            MPI_Bcast(global_col_buffer.data(), n, MPI_INT, last_col_owner, MPI_COMM_WORLD);

            // Each process updates its block of the matrix D by comparing the current distance D[i][j]
            // with the potential shorter path col_buffer[i] + row_buffer[j]. 
            // If the new path is shorter, it updates D[i][j].

            if (rank == 2) {
                std::cout << "iteration " << k << " process " << rank << " has: \n";
                print_matrix(local_matrix);
            }
            
            update_local_matrix(local_matrix, global_row_buffer, global_col_buffer, grid_row, grid_col);

        }
    }
    int calculate_matrix_dimension(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
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
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1; // In case of an error
    }
    void read_local_block(std::vector<std::vector<int>>& local_matrix) {
        std::ifstream file(input_file_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int sqrt_p = std::sqrt(size);
        int row_block = rank / sqrt_p;
        int col_block = rank % sqrt_p;
        int start_row = row_block * m_block_size;
        int start_col = col_block * m_block_size;

        int value;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file >> value;
                if (i >= start_row && i < start_row + m_block_size && j >= start_col && j < start_col + m_block_size) {
                    local_matrix[i - start_row][j - start_col] = value;
                }
            }
        }
    }
    bool should_send_row(int& k, int& sqrt_p, int& grid_row) {
        int extra_range = (grid_row == sqrt_p - 1 ? 1 : 0);
        return (int(rank / sqrt_p) < k && k < int(rank / sqrt_p) + m_block_size + extra_range);
    }
    bool should_send_column(int& k, int& sqrt_p, int& grid_col, int& rank) {
        int extra_range = (grid_col == sqrt_p - 1 ? 1 : 0);
        return ((rank % sqrt_p) < k && (k < rank % sqrt_p + m_block_size) + extra_range);
    }
public:
    void execute() {
        MPI_Comm grid_comm;
        initialize_grid(grid_comm);

        std::vector<std::vector<int>> matrix;
        if (rank == MPI_ROOT) {
            n = calculate_matrix_dimension(input_file_path);
        }

        MPI_Bcast(&n, 1, MPI_INT, MPI_ROOT, MPI_COMM_WORLD);
        m_block_size = n / dims[0];

        std::vector<std::vector<int>> local_matrix(m_block_size, std::vector<int>(m_block_size));
        read_local_block(local_matrix);

        floyd_all_pairs_parallel(local_matrix, n, grid_comm);

        if (true) {
            std::cout << "Process " << rank << " has submatrix after:\n";
            print_matrix(local_matrix);
        }

        std::vector<int> full_matrix;
        if (rank == MPI_ROOT) {
            full_matrix.resize(n * n);
        }
        gather_matrix(local_matrix, full_matrix, m_block_size, n);
        if (rank == MPI_ROOT) {
            print_full_matrix(full_matrix, n);
            write_matrix_to_file(full_matrix, n);
        }

        MPI_Comm_free(&grid_comm);
    }

private:
    int rank, size, n;
    int dims[2];
    std::string input_file_path;

    void initialize_grid(MPI_Comm& grid_comm) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);
        int periods[2] = { 0, 0 };
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    }

    void print_matrix(const std::vector<std::vector<int>>& matrix) {
        std::stringstream stream;
        for (const auto& row : matrix) {
            for (const auto& elem : row) {
                stream << elem << " ";
            }
            stream << std::endl;
        }
        std::cout << stream.str();
    }

    void print_vector(std::vector<int> const& input)
    {
        for (int i = 0; i < input.size(); i++) {
            std::cout << input.at(i) << ' ';
        }
        std::cout << "\n";
    }

    void read_matrix_from_file(std::vector<std::vector<int>>& matrix, const std::string& filename) {
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

    std::vector<int> create_2D_partition(const std::vector<std::vector<int>>& matrix, int block_size) {
        int num_blocks = size;
        std::vector<int> flat_matrix(matrix.size() * matrix.size());
        for (int p = 0; p < num_blocks; p++) {
            int block_row = p / block_size;
            int block_col = p % block_size;
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    flat_matrix[p * n + i * block_size + j] =
                        matrix[block_row * block_size + i][block_col * block_size + j];
                }
            }
        }
        return flat_matrix;
    }

    std::vector<std::vector<int>> reconstruct_matrix(const std::vector<int>& flat_matrix, int block_size) {
        std::vector<std::vector<int>> matrix(block_size, std::vector<int>(block_size));
        for (int i = 0; i < block_size; ++i) {
            std::copy(flat_matrix.begin() + i * block_size,
                flat_matrix.begin() + (i + 1) * block_size,
                matrix[i].begin());
        }
        return matrix;
    }

    void update_local_matrix(std::vector<std::vector<int>>& local_matrix,
        const std::vector<int>& global_row_buffer,
        const std::vector<int>& global_col_buffer, 
        int&grid_row, int& grid_col) {
        for (int i = 0; i < local_matrix.size(); i++) {
            for (int j = 0; j < local_matrix[i].size(); j++) {
                if (local_matrix[i][j] > global_col_buffer[j + grid_col * m_block_size] + global_row_buffer[i + grid_row * m_block_size]) {
                    local_matrix[i][j] = global_col_buffer[j + grid_col * m_block_size] + global_row_buffer[i * grid_row];
                    m_log_stream << "\nprocess " << rank << " updated element " << i << "," << j << " to " << global_col_buffer[j + grid_col * m_block_size] << " + " << global_row_buffer[i + grid_row * m_block_size] << " [" << global_col_buffer[j + grid_col * m_block_size] + global_row_buffer[i + grid_row * m_block_size] << "]\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                }
            }
        }
    }

    void gather_matrix(const std::vector<std::vector<int>>& local_matrix, std::vector<int>& full_matrix, int block_size, int n) {
        std::vector<int> displacements(size, 0);
        std::vector<int> recvcounts(size, block_size * block_size);
        int sqrt_p = static_cast<int>(sqrt(size));  // Assuming size is a perfect square

        // Flatten the local submatrix into a 1D vector
        std::vector<int> flat_local_matrix(block_size * block_size);
        for (int i = 0; i < block_size; ++i) {
            std::copy(local_matrix[i].begin(), local_matrix[i].end(), flat_local_matrix.begin() + i * block_size);
        }

        // Gather the submatrices from all processes into the full matrix
        MPI_Gather(flat_local_matrix.data(), block_size * block_size, MPI_INT,
            rank == MPI_ROOT ? &full_matrix[rank * n] : nullptr,
            block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }


    void print_full_matrix(const std::vector<int>& matrix, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i * n + j] == INF) {
                    std::cout << "INF ";
                }
                else {
                    std::cout << matrix[i * n + j] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    void write_matrix_to_file(const std::vector<int>& matrix, int n) {
        std::ofstream out_file(input_file_path + "_result.txt");
        if (!out_file.is_open()) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i * n + j] == INF) {
                    out_file << "INF ";
                }
                else {
                    out_file << matrix[i * n + j] << " ";
                }
            }
            out_file << std::endl;
        }
        out_file.close();
    }
};