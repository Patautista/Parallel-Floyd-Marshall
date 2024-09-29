# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(FloydWarshall LANGUAGES CXX)

# Find the MPI package
find_package(MPI REQUIRED)

# Set C++ standard (optional, adjust as needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(FloydWarshallMPI "src/FloydWarshallMPI.cpp" "src/implementations/ParallelFloydWarshall.cpp" "src/implementations/SerialFloydWarshall.cpp" "include/Logger.h" "include/Json.hpp" "include/Matrix.h" "src/options/FloydOptions.cpp" "Dockerfile")
add_executable(GraphGenerator "src/GraphGenerator.cpp")

# Link MPI to the project
target_link_libraries(FloydWarshallMPI MPI::MPI_CXX)

# Copy samples directory to the build directory
file(COPY samples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
