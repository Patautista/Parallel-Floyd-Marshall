#include <string>
#include <iostream>
#include <fstream>
#include <Json.h>
#pragma once

class FloydOptions {
public:
    std::string LogLevel;
    std::string InputPath;
    std::string LogOutput;
    std::string Method;

    // Method to load options from a JSON file
    bool load_from_file(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return false;
        }

        try {
            nlohmann::json j;
            file >> j;

            // Parse the contents into the class members
            LogLevel = j.value("LogLevel", "INFO");
            InputPath = j.value("InputPath", "");
            LogOutput = j.value("LogOutput", "floyd.log");
            Method = j.value("Method", "Serial");

        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing the JSON file: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    // Method to print a sample of the expected options.json file
    static void print_sample() {
        std::cout << R"({
  "LogLevel": "INFO", // DEBUG, INFO, WARNING, ERROR
  "InputPath": "", // Path to adjacency matrix (text)
  "LogOutput": "floyd.log", // Logger output
  "Method": "Serial" // Serial, Parallel
})" << std::endl;
    }
};