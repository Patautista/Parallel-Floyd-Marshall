#include "SerialFloydWarshall.cpp"
#include <ParallelFloydWarshall.cpp>
#include "FloydOptions.cpp"
#include <mpi.h>
#include <filesystem>

using namespace std::filesystem;

#define MPI_ROOT 0


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FloydOptions options;
    std::string serialized_options;

    const std::string file_path = current_path().string() + "/options.json";

    if (rank == MPI_ROOT) {
        // Root process: read and serialize options
        if (!options.load_from_file(file_path)) {
            std::cerr << "Warning: '" << file_path << "' not found or is invalid. Please ensure the file exists with the correct structure." << "Currently looking in " << current_path().string() << std::endl;
            FloydOptions::print_sample();
            MPI_Abort(MPI_COMM_WORLD, 1);  // Exit with error code
        }
        serialized_options = options.serialize(); // Serialize options to string
    }

    // Broadcast the size of the serialized options string
    int option_size = serialized_options.size();
    MPI_Bcast(&option_size, 1, MPI_INT, MPI_ROOT, MPI_COMM_WORLD);

    // Broadcast the serialized options string
    std::vector<char> buffer(option_size);
    if (rank == MPI_ROOT) {
        std::copy(serialized_options.begin(), serialized_options.end(), buffer.begin());
    }

    MPI_Bcast(buffer.data(), option_size, MPI_CHAR, MPI_ROOT, MPI_COMM_WORLD);

    if (rank != MPI_ROOT) {
        // Non-root processes: deserialize options
        std::string received_data(buffer.begin(), buffer.end());
        options = FloydOptions::deserialize(received_data);
    }

    // Proceed with using the options object on all processes
    Logger& logger = Logger::getInstance();
    logger.enableFileLogging(options.LogOutput);
    logger.setLogLevel(logger.stringToLogLevel(options.LogLevel));

    // Success: You can use the options here
    if (rank == MPI_ROOT) {
        std::cout << "LogLevel: " << options.LogLevel << std::endl;
        std::cout << "InputPath: " << options.InputPath << std::endl;
        std::cout << "LogOutput: " << options.LogOutput << std::endl;
        std::cout << "Method: " << options.Method << std::endl;
    }

    if (options.Method == "Parallel") {
        ParallelFloydWarshall floyd(options);
        floyd.execute();
    }
    else {
        if (rank == MPI_ROOT) {
            SerialFloydWarshall floyd(options);
            floyd.execute();
        }
        MPI_Finalize();
    }

    if (rank == MPI_ROOT) {
        logger.info("\nProgram finished successfully.\n");
    }

    return 0;
}