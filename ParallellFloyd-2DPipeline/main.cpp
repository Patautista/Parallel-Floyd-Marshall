#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "ParallellFloydWarshall.cpp"

int main(int argc, char** argv) {
    Logger& logger = Logger::getInstance();
    logger.enableFileLogging("app.log");
    logger.setLogLevel(LogLevel::INFO);

    ParallelFloydWarshall floyd(argc, argv);
    floyd.execute();
    return 0;
}


