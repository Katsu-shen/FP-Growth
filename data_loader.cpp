#include "data_loader.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

// clean data before we start mining data
std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(start, end - start);
}

// split one csv row by commas
std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> values;
    std::string token;
    std::stringstream stream(line);

    while (std::getline(stream, token, ',')) {
        values.push_back(trim(token));
    }

    return values;
}

// check if a value can be parsed as a number
bool parseDouble(const std::string& value, double& parsed) {
    std::stringstream stream(value);
    stream >> parsed;
    stream >> std::ws;
    return stream.eof() && !value.empty();
}

// load csv rows where the last value is the class label
std::vector<RawLabeledTransaction> loadRawLabeledTransactions(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    std::vector<RawLabeledTransaction> transactions;
    std::size_t expectedFeatureCount = 0;
    std::string line;
    int lineNumber = 0;

    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::vector<std::string> values = splitCsvLine(line);
        if (values.size() < 2) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " needs at least one feature and one class label.");
        }

        std::string classLabel = values.back();
        values.pop_back();

        if (classLabel.empty()) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has an empty class label.");
        }

        if (expectedFeatureCount == 0) {
            expectedFeatureCount = values.size();
        } else if (values.size() != expectedFeatureCount) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has a different number of features.");
        }

        transactions.push_back({values, classLabel});
    }

    if (transactions.empty()) {
        throw std::runtime_error("Input file has no transactions.");
    }

    return transactions;
}
