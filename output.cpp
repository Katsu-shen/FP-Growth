#include "output.hpp"

#include <algorithm>
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
    int transactionCount,
    std::size_t maxRules
) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Transactions: " << transactionCount << "\n\n";
    std::cout << "Classification rules\n";

    if (rules.empty()) {
        std::cout << "  No rules met the support and confidence thresholds.\n";
        return;
    }

    std::size_t rulesToPrint = rules.size();
    if (maxRules != 0) {
        rulesToPrint = std::min(rules.size(), maxRules);
    }

    for (std::size_t index = 0; index < rulesToPrint; ++index) {
        const ClassRule& rule = rules[index];
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

// parse a positive count for a rule limit or pattern length limit
std::size_t parsePositiveCount(const std::string& value, const std::string& name) {
    std::size_t parsed = 0;
    std::stringstream stream(value);
    stream >> parsed;
    stream >> std::ws;

    if (!stream.eof() || parsed == 0) {
        throw std::runtime_error(name + " must be a positive whole number.");
    }

    return parsed;
}

// explain how to run the program if the arguments are wrong
void printUsage(const char* programName) {
    std::cerr
        << "Usage: " << programName
        << " <data.csv> <min_support> <min_confidence> [--class <label>]"
        << " [--max-length <count>] [--top <count>] [--backend <cpu|cuda>]"
        << " [--header|--no-header]\n"
        << "Example: " << programName
        << " bank+marketing/bank_clean.csv 0.05 0.7 --class yes --max-length 3 --top 50\n";
}
