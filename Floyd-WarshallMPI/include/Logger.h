#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <mutex>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    // Singleton instance for global access
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    // Set log level
    void setLogLevel(LogLevel level) {
        logLevel = level;
    }

    // Enable file logging, with clearing file content when created
    void enableFileLogging(const std::string& filename) {
        fileStream.open(filename, std::ios::out);  // Clears the file when opened
        if (!fileStream.is_open()) {
            std::cerr << "Error opening log file!" << std::endl;
        }
    }

    LogLevel stringToLogLevel(const std::string& level) {
        if (level == "DEBUG") {
            return LogLevel::DEBUG;
        }
        else if (level == "INFO") {
            return LogLevel::INFO;
        }
        else if (level == "WARNING") {
            return LogLevel::WARNING;
        }
        else {
            return LogLevel::ERROR;
        }
    }

    // Log a message with a specific log level
    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mtx);
        if (level >= logLevel) {
            std::string formattedMessage = formatMessage(level, message);

            // Output to console
            std::cout << formattedMessage << std::endl;

            // Output to file if enabled
            if (fileStream.is_open()) {
                fileStream << formattedMessage << std::endl;
            }
        }
    }

    // Convenience methods for different log levels
    void debug(const std::string& message) {
        log(LogLevel::DEBUG, message);
    }

    void info(const std::string& message) {
        log(LogLevel::INFO, message);
    }

    void warning(const std::string& message) {
        log(LogLevel::WARNING, message);
    }

    void error(const std::string& message) {
        log(LogLevel::ERROR, message);
    }

private:
    LogLevel logLevel = LogLevel::INFO;
    std::ofstream fileStream;
    std::mutex mtx;  // Ensures thread safety

    // Private constructor for Singleton pattern
    Logger() = default;

    // Disable copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Format log messages
    std::string formatMessage(LogLevel level, const std::string& message) {
        std::stringstream ss;
        //ss << "[" << getCurrentTime() << "] ";
        ss << "[" << logLevelToString(level) << "] ";
        ss << message;
        return ss.str();
    }

    // Convert log level enum to string
    std::string logLevelToString(LogLevel level) {
        switch (level) {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
        }
    }

    // Get current time as string
    std::string getCurrentTime() {
        std::time_t now = std::time(nullptr);
        std::tm localTime;
        //localtime_s(&localTime, &now);  // Safe version of localtime
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime);
        return buffer;
    }
};
