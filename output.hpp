#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <string>
#include <vector>

#include "common.hpp"

std::string formatItemset(const Itemset& itemset);
void printClassificationRules(
    const std::vector<ClassRule>& rules,
    int transactionCount
);
double parseRatio(const std::string& value, const std::string& name);
void printUsage(const char* programName);

#endif
