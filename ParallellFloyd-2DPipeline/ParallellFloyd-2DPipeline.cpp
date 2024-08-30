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
    ParallelFloydWarshall floyd(argc, argv);
    floyd.execute();
    return 0;
}


