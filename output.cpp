#include "output.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// print one itemset with commas between items
std::string formatItemset(const Itemset& itemset) {
    std::ostringstream output;
    output << "{";

    for (std::size_t index = 0; index < itemset.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << itemset[index];
    }

    output << "}";
    return output.str();
}

// print class rules with support ratio, confidence, and count
void printClassificationRules(
    const std::vector<ClassRule>& rules,
    int transactionCount
) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Transactions: " << transactionCount << "\n\n";
    std::cout << "Classification rules\n";

    if (rules.empty()) {
        std::cout << "  No rules met the support and confidence thresholds.\n";
        return;
    }

    for (const ClassRule& rule : rules) {
        std::cout << "  " << std::setw(40) << std::left << formatItemset(rule.antecedent)
                  << " -> {" << rule.classLabel << "}"
                  << " support=" << rule.support
                  << " confidence=" << rule.confidence
                  << " count=" << rule.count << "\n";
    }
}

// parse min support ratio or min confidence from command line
double parseRatio(const std::string& value, const std::string& name) {
    double parsed = 0.0;
    std::stringstream stream(value);
    stream >> parsed;
    stream >> std::ws;

    if (!stream.eof() || parsed < 0.0 || parsed > 1.0) {
        throw std::runtime_error(name + " must be a number between 0 and 1.");
    }

    return parsed;
}

// explain how to run the program if the arguments are wrong
void printUsage(const char* programName) {
    std::cerr
        << "Usage: " << programName << " <data.csv> <min_support> <min_confidence>\n"
        << "Example: " << programName << " iris/iris.data 0.2 0.7\n";
}
