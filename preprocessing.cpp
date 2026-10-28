#include "preprocessing.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include "data_loader.hpp"

// missing values are skipped instead of being mined as items
bool isMissingValue(const std::string& value) {
    return value == "?";
}

// decide whether every value in each feature column is numeric
std::vector<bool> findNumericColumns(const std::vector<RawLabeledTransaction>& transactions) {
    std::vector<bool> numericColumns(transactions.front().values.size(), true);

    for (const RawLabeledTransaction& transaction : transactions) {
        for (std::size_t index = 0; index < transaction.values.size(); ++index) {
            if (isMissingValue(transaction.values[index])) {
                continue;
            }

            double parsed = 0.0;
            if (!parseDouble(transaction.values[index], parsed)) {
                numericColumns[index] = false;
            }
        }
    }

    return numericColumns;
}

// get the min and max value for every numeric column
void findNumericRanges(
    const std::vector<RawLabeledTransaction>& transactions,
    const std::vector<bool>& numericColumns,
    std::vector<double>& mins,
    std::vector<double>& maxes
) {
    mins.assign(numericColumns.size(), std::numeric_limits<double>::max());
    maxes.assign(numericColumns.size(), std::numeric_limits<double>::lowest());

    for (const RawLabeledTransaction& transaction : transactions) {
        for (std::size_t index = 0; index < transaction.values.size(); ++index) {
            if (!numericColumns[index]) {
                continue;
            }

            if (isMissingValue(transaction.values[index])) {
                continue;
            }

            double parsed = 0.0;
            parseDouble(transaction.values[index], parsed);
            mins[index] = std::min(mins[index], parsed);
            maxes[index] = std::max(maxes[index], parsed);
        }
    }
}

// turn a numeric value into low, medium, or high
std::string binNumericValue(double value, double minValue, double maxValue) {
    if (minValue == maxValue) {
        return "medium";
    }

    double width = (maxValue - minValue) / 3.0;
    double lowCutoff = minValue + width;
    double highCutoff = minValue + (2.0 * width);

    if (value < lowCutoff) {
        return "low";
    }

    if (value < highCutoff) {
        return "medium";
    }

    return "high";
}

// convert raw csv values into fp-growth items
std::vector<LabeledTransaction> binLabeledTransactions(
    const std::vector<RawLabeledTransaction>& rawTransactions,
    const std::vector<std::string>& featureNames
) {
    if (featureNames.size() != rawTransactions.front().values.size()) {
        throw std::runtime_error("Feature name count does not match transaction width.");
    }
    std::vector<bool> numericColumns = findNumericColumns(rawTransactions);
    std::vector<double> mins;
    std::vector<double> maxes;
    findNumericRanges(rawTransactions, numericColumns, mins, maxes);

    std::vector<LabeledTransaction> transactions;

    for (const RawLabeledTransaction& rawTransaction : rawTransactions) {
        std::set<std::string> uniqueItems;

        for (std::size_t index = 0; index < rawTransaction.values.size(); ++index) {
            if (isMissingValue(rawTransaction.values[index])) {
                continue;
            }

            std::string itemName = featureNames[index] + "=";

            if (numericColumns[index]) {
                double parsed = 0.0;
                parseDouble(rawTransaction.values[index], parsed);
                itemName += binNumericValue(parsed, mins[index], maxes[index]);
            } else {
                itemName += rawTransaction.values[index];
            }

            uniqueItems.insert(itemName);
        }

        transactions.push_back({
            Itemset(uniqueItems.begin(), uniqueItems.end()),
            rawTransaction.classLabel
        });
    }

    return transactions;
}

// keep just the feature items so fp-growth does not mine class labels as items
Transactions getFeatureTransactions(const std::vector<LabeledTransaction>& labeledTransactions) {
    Transactions transactions;

    for (const LabeledTransaction& transaction : labeledTransactions) {
        transactions.push_back(transaction.items);
    }

    return transactions;
}
