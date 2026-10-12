#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <vector>

using Itemset = std::vector<std::string>;
using Transactions = std::vector<Itemset>;

// transaction plus count for conditional trees
struct WeightedTransaction {
    Itemset items;
    int count;
};

// raw csv row before numeric values are turned into bins
struct RawLabeledTransaction {
    std::vector<std::string> values;
    std::string classLabel;
};

// cleaned transaction where items are ready for fp-growth
struct LabeledTransaction {
    Itemset items;
    std::string classLabel;
};

// final classification rule
struct ClassRule {
    Itemset antecedent;
    std::string classLabel;
    int count;
    double support;
    double confidence;
};

#endif
