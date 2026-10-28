#include "data_loader.hpp"

#include <cctype>
#include <fstream>
#include <set>
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
LoadedLabeledTransactions loadRawLabeledTransactions(const std::string& path, bool hasHeader) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    LoadedLabeledTransactions loaded;
    std::size_t expectedFeatureCount = 0;
    bool headerRead = !hasHeader;
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

        if (!headerRead) {
            values.pop_back();
            std::set<std::string> uniqueNames;
            for (const std::string& name : values) {
                if (name.empty()) {
                    throw std::runtime_error("Header has an empty feature name.");
                }
                if (!uniqueNames.insert(name).second) {
                    throw std::runtime_error("Header has a duplicate feature name: " + name);
                }
            }
            loaded.featureNames = std::move(values);
            expectedFeatureCount = loaded.featureNames.size();
            headerRead = true;
            continue;
        }

        std::string classLabel = values.back();
        values.pop_back();

        if (classLabel.empty()) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has an empty class label.");
        }

        if (expectedFeatureCount == 0) {
            expectedFeatureCount = values.size();
            for (std::size_t index = 0; index < expectedFeatureCount; ++index) {
                loaded.featureNames.push_back("attr" + std::to_string(index + 1));
            }
        } else if (values.size() != expectedFeatureCount) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has a different number of features.");
        }

        loaded.transactions.push_back({values, classLabel});
    }

    if (!headerRead) {
        throw std::runtime_error("Input file has no header row.");
    }

    if (loaded.transactions.empty()) {
        throw std::runtime_error("Input file has no transactions.");
    }

    return loaded;
}
