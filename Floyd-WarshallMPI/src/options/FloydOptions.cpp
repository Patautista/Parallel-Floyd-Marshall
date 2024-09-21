#include <string>
#include <iostream>
#include <cstdlib>
#include "../../include/Json.hpp"
#pragma once

class FloydOptions {
public:
    std::string InputPath;  // Path for input data
    std::string Method;     // Computational method

    // Constructor to initialize options from environment variables
    FloydOptions() {
        char* envInputPath = std::getenv("INPUT_PATH");
        char* envMethod = std::getenv("METHOD");

        InputPath = envInputPath ? std::string(envInputPath) : "./samples/1/matrix.txt";
        Method = envMethod ? std::string(envMethod) : "Serial";
    }

    // Serialize the object's state to a JSON string
    std::string serialize() const {
        nlohmann::json j;
        j["InputPath"] = InputPath;
        j["Method"] = Method;
        return j.dump(); // Convert to a compact string representation
    }

    // Deserialize the object's state from a JSON string
    static FloydOptions deserialize(const std::string& data) {
        nlohmann::json j = nlohmann::json::parse(data);
        FloydOptions options;
        options.InputPath = j.value("InputPath", "");
        options.Method = j.value("Method", "Serial");
        return options;
    }

    // Print the current configuration settings
    void printSettings() const {
        std::cout << "Current Configuration:\n"
            << "InputPath: " << InputPath << "\n"
            << "Method: " << Method << std::endl;
    }
};