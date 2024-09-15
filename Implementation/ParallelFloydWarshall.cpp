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
#include "Matrix.h"
#include <omp.h>
#include "../FloydWarshallAllPairs/FloydOptions.cpp"

#define INF INT_MAX
#define MPI_ROOT 0

class ParallelFloydWarshall {
public:
    ParallelFloydWarshall(FloydOptions& options) : m_options(options){
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    ~ParallelFloydWarshall() {
        MPI_Finalize();
    }

private:
    FloydOptions m_options;
    int m_block_size;
    Logger& m_logger = Logger::getInstance();
    std::stringstream m_log_stream;

    void floyd_all_pairs_parallel(std::vector<std::vector<int>>& local_matrix, int n, MPI_Comm& comm, int sqrt_p) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<int> global_row_buffer(n);
        std::vector<int> global_col_buffer(n);

        int process_grid_col = rank % sqrt_p;
        int process_grid_row = int(rank / m_block_size);

        for (int k = 0; k < n; k++) {

            int k_grid_index = int(k / m_block_size);
            int last_row_owner = (k_grid_index * sqrt_p) + sqrt_p - 1;

            if (should_send_row(k, sqrt_p, process_grid_row)) {
                int local_row_index = k % m_block_size;

                #pragma omp parallel for
                for (int i = 0; i < m_block_size; i++) {
                    global_row_buffer[(process_grid_col * m_block_size) + i] = local_matrix[local_row_index][i];
                }

                if (process_grid_col > 0) {
                    int coords[2];
                    MPI_Cart_coords(comm, rank, 2, coords);
                    coords[1]--;
                    int rec_partner;
                    MPI_Cart_rank(comm, coords, &rec_partner);

                    std::vector<int> temp(m_block_size * process_grid_col);

                    m_log_stream << "\niteration " << k << " : " << rank << " receives row from " << rec_partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();

                    MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
                }

                int coords[2];
                MPI_Cart_coords(comm, rank, 2, coords);
                coords[1]++;
                if (rank != last_row_owner) {
                    int partner;
                    MPI_Cart_rank(comm, coords, &partner);
                    m_log_stream << "\niteration " << k << " : " << rank << " sends row to " << partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    MPI_Send(global_row_buffer.data(), m_block_size * (process_grid_col + 1), MPI_INT, partner, 1, MPI_COMM_WORLD);
                }
            }

            MPI_Bcast(global_row_buffer.data(), n, MPI_INT, last_row_owner, MPI_COMM_WORLD);
            
            // sets column broadcaster index
            int last_col_owner = size - sqrt_p + k_grid_index;

            if (should_send_column(k, sqrt_p, process_grid_col)) {

                if (process_grid_row > 0) {
                    int rec_partner = rank - m_block_size;
                    std::vector<int> temp(m_block_size * process_grid_row);
                    m_log_stream << "\niteration " << k << " : " << rank << " receives column from " << rec_partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::copy(temp.begin(), temp.end(), global_col_buffer.begin());
                }

                int local_col_index = k % m_block_size;
                #pragma omp parallel for
                for (int i = 0; i < m_block_size; i++) {
                    global_col_buffer[(process_grid_row * m_block_size) + i] = local_matrix[i][local_col_index];
                }
                int partner = rank + m_block_size;
                if (rank != last_col_owner) {
                    m_log_stream << "\niteration " << k << " : " << rank << " sends column to " << partner << "\n";
                    m_logger.debug(m_log_stream.str());
                    m_log_stream.flush();
                    MPI_Send(global_col_buffer.data(), m_block_size * (process_grid_row + 1), MPI_INT, partner, 2, MPI_COMM_WORLD);
                }
            }
            MPI_Bcast(global_col_buffer.data(), n, MPI_INT, last_col_owner, MPI_COMM_WORLD);
            
            update_local_matrix(local_matrix, global_row_buffer, global_col_buffer, process_grid_row, process_grid_col, k);
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
    void read_matrix_from_file_parallel(std::vector<std::vector<int>>& local_matrix, int sqrt_p) {
        std::ifstream file(m_options.InputPath);
        if (!file.is_open()) {
            std::cerr << "Unable to open file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int row_block = rank / sqrt_p;
        int col_block = rank % sqrt_p;
        int start_row = row_block * m_block_size;
        int start_col = col_block * m_block_size;

        int value;
        #pragma omp parallel for
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
        int top_limit = grid_row * m_block_size;
        int bottom_limit = top_limit + m_block_size - 1;
        return k >= top_limit && k <= bottom_limit;
    }
    bool should_send_column(int& k, int& sqrt_p, int& grid_col) {
        int left_limit = grid_col * m_block_size;
        int right_limit = left_limit + m_block_size - 1;
        return k>= left_limit && k <= right_limit;
    }
public:
    void execute() {
        MPI_Comm grid_comm;
        initialize_grid(grid_comm);

        std::vector<std::vector<int>> matrix;
        if (rank == MPI_ROOT) {
            n = calculate_matrix_dimension(m_options.InputPath);
        }

        MPI_Bcast(&n, 1, MPI_INT, MPI_ROOT, MPI_COMM_WORLD);

        int sqrt_p = static_cast<int>(sqrt(size));
        m_block_size = int(n / sqrt_p);

        std::vector<std::vector<int>> local_matrix(m_block_size, std::vector<int>(m_block_size));
        read_matrix_from_file_parallel(local_matrix, sqrt_p);

        floyd_all_pairs_parallel(local_matrix, n, grid_comm, sqrt_p);

        std::vector<int> full_matrix;
        if (rank == MPI_ROOT) {
            full_matrix.resize(n * n);
        }
        gather_matrix(local_matrix, full_matrix, m_block_size, n, sqrt_p);
        if (rank == MPI_ROOT) {
            write_flat_matrix_to_file(full_matrix, n, m_block_size, sqrt_p, m_options.InputPath + "_result_parallel.txt");
        }

        MPI_Comm_free(&grid_comm);
    }

private:
    int rank, size, n;
    int dims[2];

    void initialize_grid(MPI_Comm& grid_comm) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);
        int periods[2] = { 0, 0 };
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    }

    void update_local_matrix(std::vector<std::vector<int>>& local_matrix, const std::vector<int>& global_row_buffer,
        const std::vector<int>& global_col_buffer, int&grid_row, int& grid_col, int k) {
        #pragma omp parallel for
        for (int i = 0; i < local_matrix.size(); i++) {
            for (int j = 0; j < local_matrix[i].size(); j++) {
                int row_buffer_index = i + grid_row * m_block_size;
                int col_buffer_index = j + grid_col * m_block_size;
                if (local_matrix[i][j] > global_col_buffer[row_buffer_index] + global_row_buffer[col_buffer_index]) {
                    local_matrix[i][j] = global_col_buffer[row_buffer_index] + global_row_buffer[col_buffer_index];
                }
            }
        }
    }

    void gather_matrix(const std::vector<std::vector<int>>& local_matrix, std::vector<int>& full_matrix, int block_size, int n, int sqrt_p) {
        std::vector<int> displacements(size, 0);
        std::vector<int> recvcounts(size, block_size * block_size);

        // Flatten the local submatrix into a 1D vector
        std::vector<int> flat_local_matrix(block_size * block_size);
        for (int i = 0; i < block_size; ++i) {
            std::copy(local_matrix[i].begin(), local_matrix[i].end(), flat_local_matrix.begin() + i * block_size);
        }

        // Gather the submatrices from all processes into the full matrix
        MPI_Gather(flat_local_matrix.data(), block_size * block_size, MPI_INT,
            rank == MPI_ROOT ? &full_matrix[rank * n] : nullptr,
            block_size * block_size, MPI_INT, MPI_ROOT, MPI_COMM_WORLD);
    }

};