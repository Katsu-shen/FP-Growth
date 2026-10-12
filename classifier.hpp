#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <map>
#include <vector>

#include "common.hpp"

bool isSubset(const Itemset& pattern, const Itemset& transaction);
std::vector<ClassRule> generateClassificationRules(
    const std::map<Itemset, int>& frequentItemsets,
    const std::vector<LabeledTransaction>& transactions,
    double minConfidence
);

#endif
