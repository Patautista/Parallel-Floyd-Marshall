#include <string>
#include <iostream>
#include <fstream>
#include "../../include/Json.hpp"
#pragma once

class FloydOptions {
public:
    std::string LogLevel;
    std::string InputPath;
    std::string LogOutput;
    std::string Method;

    // Method to load options from a JSON file
    bool load(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cout << "Options file not found. \n";
            return false;
        }

        std::cout << "Options file found. \n";

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

    std::string serialize() const {
        nlohmann::json j;
        j["LogLevel"] = LogLevel;
        j["InputPath"] = InputPath;
        j["LogOutput"] = LogOutput;
        j["Method"] = Method;
        return j.dump(); // Convert to a compact string representation
    }

    // Deserialize the options from a JSON string
    static FloydOptions deserialize(const std::string& data) {
        nlohmann::json j = nlohmann::json::parse(data);
        FloydOptions options;
        options.LogLevel = j.value("LogLevel", "INFO");
        options.InputPath = j.value("InputPath", "");
        options.LogOutput = j.value("LogOutput", "floyd.log");
        options.Method = j.value("Method", "Serial");
        return options;
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