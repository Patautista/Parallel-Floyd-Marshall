#include "SerialFloydWarshall.cpp"
#include <ParallelFloydWarshall.cpp>
#include "FloydOptions.cpp"
#include <mpi.h>

#define MPI_ROOT 0


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FloydOptions options;
    const std::string file_path = "options.json";

    // Try to load options from file
    if (!options.load_from_file(file_path)) {
        if (rank == MPI_ROOT) {
            std::cerr << "Warning: '" << file_path << "' not found or is invalid. Please ensure the file exists with the correct structure." << std::endl;
            FloydOptions::print_sample();
        }
        return 1;  // Exit with error code
    }

    Logger& logger = Logger::getInstance();
    logger.enableFileLogging(options.LogOutput);
    // Convert log level string to enum
    
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
        SerialFloydWarshall floyd(options);
        floyd.execute();
    }
    return 0;
}