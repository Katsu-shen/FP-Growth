#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>

#include "common.hpp"

std::string trim(const std::string& value);
std::vector<std::string> splitCsvLine(const std::string& line);
bool parseDouble(const std::string& value, double& parsed);
std::vector<RawLabeledTransaction> loadRawLabeledTransactions(const std::string& path);

#endif
