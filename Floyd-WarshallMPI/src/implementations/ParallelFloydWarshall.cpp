#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <format>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <omp.h>
#include "../../include/Logger.h"
#include "../options/FloydOptions.cpp"
#include "../../include/Matrix.h"

#define INF INT_MAX
#define MPI_ROOT 0

class ProcessData {
    public:
        int row_count;
        int col_count;

        int row_start;
        int row_end;
        int column_start;
        int column_end;
};

class ParallelFloydWarshall {
public:
    ParallelFloydWarshall(const FloydOptions& options) : m_options(options){
        
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

    std::vector<ProcessData> process_datas;

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

        process_datas.resize(size);

        for (int p = 0; p < size; ++p) {
            auto [p_row_size, p_col_size] = get_process_block_sizes(p, n, sqrt_p);
            int p_row_block = p / sqrt_p;
            int p_col_block = p % sqrt_p;
            process_datas[p].row_count = p_row_size;
            process_datas[p].col_count = p_col_size;
            
            process_datas[p].row_start = (p_row_block < remainder) ? p_row_block * (m_base_block_size + remainder) : p_row_block * m_base_block_size + remainder;
            process_datas[p].row_end = process_datas[p].row_start + process_datas[p].col_count - 1;
            
            process_datas[p].column_start = (p_col_block < remainder) ? p_col_block * (m_base_block_size + 1) : p_col_block * m_base_block_size + remainder;
            process_datas[p].column_end = process_datas[p].column_start + process_datas[p].row_count - 1;
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

            int k_grid_index = k < m_base_block_size + (n % sqrt_p) ? 0 : int((k - (n % sqrt_p)) / m_base_block_size);

            int last_row_owner = (k_grid_index * sqrt_p) + sqrt_p - 1;

            if (should_send_row(k, sqrt_p)) {
                int local_row_index = k - m_row_start;

                #pragma omp parallel for
                for (int j = 0; j < m_col_count; j++) {
                    global_row_buffer[(m_column_start)+j] = local_matrix[local_row_index][j];
                }

                if (process_grid_col > 0) {
                    int rec_partner = rank - 1;

                    std::vector<int> temp(m_column_start);

                    MPI_Recv(temp.data(), m_column_start, MPI_INT, rec_partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::copy(temp.begin(), temp.end(), global_row_buffer.begin());
                }

                if (rank != last_row_owner && rank + 1 < size) {
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

                int local_col_index = k - m_column_start;
                #pragma omp parallel for
                for (int i = 0; i < m_row_count; i++) {
                    global_col_buffer[(m_row_start) + i] = local_matrix[i][local_col_index];
                }
                if (rank != last_col_owner && rank + sqrt_p < size){
                    int partner = rank + sqrt_p;
                    MPI_Send(global_col_buffer.data(), m_row_start + m_row_count, MPI_INT, partner, 2, MPI_COMM_WORLD);
                }
            }
            MPI_Bcast(global_col_buffer.data(), n, MPI_INT, last_col_owner, MPI_COMM_WORLD);

            update_local_matrix(local_matrix, global_row_buffer, global_col_buffer, process_grid_row, process_grid_col, k);
        }
    }
    void write_matrix_to_file_parallel(std::vector<std::vector<int>>& local_matrix, int sqrt_p, MPI_Comm& comm, const std::string& file_path) {

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

            broadcast_in_loop(k, rank, size, comm, local_matrix, global_row_buffer, sqrt_p, true);

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

    void broadcast_in_loop(int k, int rank, int size, MPI_Comm comm,
        std::vector<std::vector<int>>& local_matrix,
        std::vector<int>& global_row_buffer,
        int sqrt_p, bool is_row_broadcast) {

        std::vector<int> broadcast_buffer(m_base_block_size, 0);
        std::vector<int> senders = {};

        // Calculate the grid index for the row that contains 'k'
        int k_grid_index = (k < m_base_block_size + (n % sqrt_p)) ? 0 : int((k - (n % sqrt_p)) / m_base_block_size);
        int coords[] = { 0, 0 };

        // Identify the processes that contain the row 'k'
        for (int i = 0; i < size; i++) {
            MPI_Cart_coords(comm, i, 2, coords);
            if ((is_row_broadcast && coords[0] == k_grid_index) || (!is_row_broadcast && coords[1] == k_grid_index)) {
                senders.push_back(i);
            }
        }

        // Broadcast the row 'k' from each sender process and update the global row buffer
        for (int i = 0; i < senders.size(); i++) {
            if (rank == senders[i]) {
                int start = is_row_broadcast ? m_row_start : m_column_start;
                int k_row_in_local_matrix = k - start;

                #pragma omp parallel for
                for (int j = 0; j < process_datas[senders[i]].row_count; j++) {
                    broadcast_buffer[j] = local_matrix[k_row_in_local_matrix][j];
                }
            }

            // Broadcast the row buffer from the current sender to all processes
            MPI_Bcast(broadcast_buffer.data(), process_datas[senders[i]].row_count, MPI_INT, senders[i], MPI_COMM_WORLD);

            // Copy the broadcasted row into the appropriate position in the global row buffer
            int offset = is_row_broadcast ? process_datas[senders[i]].column_start : process_datas[senders[i]].row_start;
            std::copy(broadcast_buffer.begin(), broadcast_buffer.end(),
                global_row_buffer.begin() + offset);
        }
    }


    bool should_send_row(int& k, int& sqrt_p) {
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

        //floyd_all_pairs_parallel(local_matrix, n, grid_comm, sqrt_p);

        write_matrix_to_file_parallel(local_matrix, sqrt_p, grid_comm, m_options.InputPath + "_result_parallel.txt");

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
};