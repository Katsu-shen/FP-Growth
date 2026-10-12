#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <string>
#include <vector>

#include "common.hpp"

bool isMissingValue(const std::string& value);
std::vector<bool> findNumericColumns(const std::vector<RawLabeledTransaction>& transactions);
void findNumericRanges(
    const std::vector<RawLabeledTransaction>& transactions,
    const std::vector<bool>& numericColumns,
    std::vector<double>& mins,
    std::vector<double>& maxes
);
std::string binNumericValue(double value, double minValue, double maxValue);
std::vector<LabeledTransaction> binLabeledTransactions(
    const std::vector<RawLabeledTransaction>& rawTransactions
);
Transactions getFeatureTransactions(const std::vector<LabeledTransaction>& labeledTransactions);

#endif
