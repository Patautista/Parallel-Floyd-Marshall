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
    int m_base_block_size;
    int m_row_count;
    int m_col_count;

    int m_row_start;
    int m_row_end;
    int m_column_start;
    int m_column_end;

    std::vector<int> m_row_block_sizes;
    std::vector<int> m_col_block_sizes;

    Logger& m_logger = Logger::getInstance();
    std::stringstream m_log_stream;

    void init_process_data(int sqrt_p) {
        m_base_block_size = int(n / sqrt_p);
        auto [a, b] = get_process_block_sizes(rank, n, sqrt_p);
        m_row_count = a;
        m_col_count = b;

        int row_block = rank / sqrt_p;
        int col_block = rank % sqrt_p;
        int remainder = n % sqrt_p;

        if (rank == MPI_ROOT) {
            m_row_block_sizes.resize(size);
            m_col_block_sizes.resize(size);

            for (int p = 0; p < size; ++p) {
                auto [p_row_size, p_col_size] = get_process_block_sizes(p, n, sqrt_p);
                m_row_block_sizes[p] = p_row_size;
                m_col_block_sizes[p] = p_col_size;
            }
        }

        m_row_start = (row_block < remainder) ? row_block * (m_base_block_size + 1) : row_block * m_base_block_size + remainder;
        m_row_end = m_row_start + m_col_count - 1;
        m_column_start = (col_block < remainder) ? col_block * (m_base_block_size + 1) : col_block * m_base_block_size + remainder;
        m_column_end = m_column_start + m_row_count - 1;
    }

    void floyd_all_pairs_parallel(std::vector<std::vector<int>>& local_matrix, int n, MPI_Comm& comm, int sqrt_p) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<int> global_row_buffer(n);
        std::vector<int> global_col_buffer(n);

        int process_grid_col = rank % sqrt_p;
        int process_grid_row = int(rank / sqrt_p);

        for (int k = 0; k < n; k++) {

            int k_grid_index = k < m_base_block_size + (n % sqrt_p) ? 0 : int((k - 1) / m_base_block_size);

            int last_row_owner = (k_grid_index * sqrt_p) + sqrt_p - 1;

            if (should_send_row(k, sqrt_p, process_grid_row)) {
                int local_row_index = k > m_base_block_size ? k % m_row_start : k;

                #pragma omp parallel for
                for (int i = 0; i < m_col_count; i++) {
                    global_row_buffer[(m_column_start)+i] = local_matrix[local_row_index][i];
                }

                if (process_grid_col > 0) {
                    int rec_partner = rank - 1;

                    std::vector<int> temp(m_column_start);

                    MPI_Recv(temp.data(), m_column_start, MPI_INT, rec_partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
                }

                if (rank != last_row_owner) {
                    int partner = rank + 1;
                    MPI_Send(global_row_buffer.data(), m_column_start + m_col_count, MPI_INT, partner, 1, MPI_COMM_WORLD);
                }
            }

            MPI_Bcast(global_row_buffer.data(), n, MPI_INT, last_row_owner, MPI_COMM_WORLD);
            
            // sets column broadcaster index
            int last_col_owner = size + k_grid_index - sqrt_p;

            if (should_send_column(k, sqrt_p, process_grid_col)) {

                if (process_grid_row > 0) {
                    int rec_partner = rank - sqrt_p;
                    std::vector<int> temp(m_row_start);
                    MPI_Recv(temp.data(), temp.size(), MPI_INT, rec_partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::copy(temp.begin(), temp.end(), global_col_buffer.begin());
                }

                int local_col_index = k > m_base_block_size ? k % m_column_start : k;
                #pragma omp parallel for
                for (int i = 0; i < m_row_count; i++) {
                    global_col_buffer[(m_row_start) + i] = local_matrix[i][local_col_index];
                }
                if (rank != last_col_owner) {
                    int partner = rank + sqrt_p;
                    MPI_Send(global_col_buffer.data(), m_row_start + m_row_count, MPI_INT, partner, 2, MPI_COMM_WORLD);
                }
            }
            MPI_Bcast(global_col_buffer.data(), n, MPI_INT, last_col_owner, MPI_COMM_WORLD);

            update_local_matrix(local_matrix, global_row_buffer, global_col_buffer, process_grid_row, process_grid_col, k);
        }
    }
    void write_matrix_to_file_parallel(std::vector<std::vector<int>>& local_matrix, int n, int sqrt_p, const std::string& file_path) {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::ofstream file;
        if (rank == MPI_ROOT) {
            file.open(file_path);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for writing!" << std::endl;
                return;
            }
        }

        std::vector<int> global_row_buffer(n);

        int process_grid_col = rank % sqrt_p;
        int process_grid_row = int(rank / m_base_block_size);

        for (int k = 0; k < n; k++) {

            int k_grid_index = k < m_base_block_size + (n % sqrt_p) ? 0 : int((k - 1) / m_base_block_size);

            int last_row_owner = (k_grid_index * sqrt_p) + sqrt_p - 1;

            if (should_send_row(k, sqrt_p, process_grid_row)) {
                int local_row_index = k > m_base_block_size ? k % m_row_start : k;

                #pragma omp parallel for
                for (int i = 0; i < m_col_count; i++) {
                    global_row_buffer[(m_column_start) + i] = local_matrix[local_row_index][i];
                }

                if (process_grid_col > 0) {
                    int rec_partner = rank - 1;

                    std::vector<int> temp(m_column_start);

                    MPI_Recv(temp.data(), m_column_start, MPI_INT, rec_partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
                }

                if (rank != last_row_owner) {
                    int partner = rank + 1;
                    MPI_Send(global_row_buffer.data(), m_column_start + m_col_count, MPI_INT, partner, 1, MPI_COMM_WORLD);
                }
            }

            MPI_Bcast(global_row_buffer.data(), n, MPI_INT, last_row_owner, MPI_COMM_WORLD);

            if (rank == MPI_ROOT) {
                for (int j = 0; j < n; j++) {
                    file << global_row_buffer[j] << " ";
                }
                file << std::endl;
            }
        }
        // Close the file in the root process
        if (rank == MPI_ROOT) {
            file.close();
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
    std::pair<int, int> get_process_block_sizes(int rank, int n, int sqrt_p) {
        // Determine the row and column position of the current process in the process grid
        int process_grid_row = rank / sqrt_p;
        int process_grid_col = rank % sqrt_p;

        // Calculate block sizes for both rows and columns
        int base_row_block_size = n / sqrt_p;
        int base_col_block_size = n / sqrt_p;

        // Extra rows/columns for uneven distribution
        int extra_rows = n % sqrt_p;
        int extra_cols = n % sqrt_p;

        int row_block_size = base_row_block_size + (process_grid_row < extra_rows ? 1 : 0);
        int col_block_size = base_col_block_size + (process_grid_col < extra_cols ? 1 : 0);

        return std::make_pair(row_block_size, col_block_size);
    }
    void read_matrix_from_file_parallel(std::vector<std::vector<int>>& local_matrix, int sqrt_p) {
        std::ifstream file(m_options.InputPath);
        if (!file.is_open()) {
            std::cerr << "Unable to open file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int value;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file >> value;
                if (i >= m_row_start && i < m_row_start + m_row_count && j >= m_column_start && j < m_column_start + m_col_count) {
                    local_matrix[i - m_row_start][j - m_column_start] = value;
                }
            }
        }
    }

    bool should_send_row(int& k, int& sqrt_p, int& grid_row) {
        return k >= m_row_start && k < m_row_start + m_row_count;
    }
    bool should_send_column(int& k, int& sqrt_p, int& grid_col) {
        return k>= m_column_start && k < m_column_start + m_col_count;
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
        init_process_data(sqrt_p);

        std::vector<std::vector<int>> local_matrix(m_row_count, std::vector<int>(m_col_count));
        read_matrix_from_file_parallel(local_matrix, sqrt_p);
        

        floyd_all_pairs_parallel(local_matrix, n, grid_comm, sqrt_p);

        std::vector<int> full_matrix;
        if (rank == MPI_ROOT) {
            full_matrix.resize(n * n);
        }
        
        //gather_matrix(local_matrix, full_matrix, sqrt_p);

        write_matrix_to_file_parallel(local_matrix, n, sqrt_p, m_options.InputPath + "_result_parallel.txt");

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
                int row_buffer_index = i + m_row_start;
                int col_buffer_index = j + m_column_start;
                if (local_matrix[i][j] > global_col_buffer[row_buffer_index] + global_row_buffer[col_buffer_index]) {
                    local_matrix[i][j] = global_col_buffer[row_buffer_index] + global_row_buffer[col_buffer_index];
                }
            }
        }
    }

    void gather_matrix(const std::vector<std::vector<int>>& local_matrix, std::vector<int>& full_matrix, int sqrt_p) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Calculate the local matrix size (number of elements in each process's submatrix)
        int local_matrix_size = m_col_count * m_row_count;

        // Flatten the local submatrix into a 1D vector
        std::vector<int> flat_local_matrix(local_matrix_size);
        for (int i = 0; i < m_row_count; ++i) {
            std::copy(local_matrix[i].begin(), local_matrix[i].end(), flat_local_matrix.begin() + i * m_col_count);
        }

        if (false) {
            std::cout << "rank :" << rank << "\n";
            print_vector(flat_local_matrix);
        }

        // Prepare recv_counts and displs arrays on the root process
        std::vector<int> recv_counts(size);
        std::vector<int> displs(size);

        if (rank == MPI_ROOT) {
            recv_counts[0] = m_row_count * m_col_count;
            int next = m_row_count * m_col_count;
            for (int i = 1; i < size; ++i) {
                // Each process contributes a local_matrix_size number of elements
                recv_counts[i] = m_row_block_sizes[i] * m_col_block_sizes[i];
                displs[i] = next;
                next += m_row_block_sizes[i] * m_col_block_sizes[i];
            }
        }

        // Use MPI_Gatherv to gather all the submatrices into the full matrix on the root process
        MPI_Gatherv(flat_local_matrix.data(), local_matrix_size, MPI_INT,
            rank == MPI_ROOT ? full_matrix.data() : nullptr,
            recv_counts.data(), displs.data(), MPI_INT,
            MPI_ROOT, MPI_COMM_WORLD);
    }


};