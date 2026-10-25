#include "classifier.hpp"

#include <algorithm>
#include <map>

// check if a pattern is inside one transaction
bool isSubset(const Itemset& pattern, const Itemset& transaction) {
    return std::includes(
        transaction.begin(),
        transaction.end(),
        pattern.begin(),
        pattern.end()
    );
}

// build classification rules from frequent itemsets
std::vector<ClassRule> generateClassificationRules(
    const std::map<Itemset, int>& frequentItemsets,
    const std::vector<LabeledTransaction>& transactions,
    double minConfidence,
    const std::string& targetClass
) {
    std::vector<ClassRule> rules;

    for (const auto& entry : frequentItemsets) {
        const Itemset& itemset = entry.first;
        int itemsetSupport = entry.second;
        std::map<std::string, int> classCounts;

        for (const LabeledTransaction& transaction : transactions) {
            if (isSubset(itemset, transaction.items)) {
                ++classCounts[transaction.classLabel];
            }
        }

        for (const auto& classCount : classCounts) {
            if (!targetClass.empty() && classCount.first != targetClass) {
                continue;
            }

            double support = static_cast<double>(classCount.second) / transactions.size();
            double confidence = static_cast<double>(classCount.second) / itemsetSupport;

            if (confidence >= minConfidence) {
                rules.push_back({
                    itemset,
                    classCount.first,
                    classCount.second,
                    support,
                    confidence
                });
            }
        }
    }

    std::sort(rules.begin(), rules.end(), [](const ClassRule& left, const ClassRule& right) {
        if (left.confidence != right.confidence) {
            return left.confidence > right.confidence;
        }

        if (left.support != right.support) {
            return left.support > right.support;
        }

        if (left.antecedent.size() != right.antecedent.size()) {
            return left.antecedent.size() < right.antecedent.size();
        }

        if (left.antecedent != right.antecedent) {
            return left.antecedent < right.antecedent;
        }

        return left.classLabel < right.classLabel;
    });

    return rules;
}
