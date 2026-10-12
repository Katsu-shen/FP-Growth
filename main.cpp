#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "classifier.hpp"
#include "common.hpp"
#include "data_loader.hpp"
#include "fp_tree.hpp"
#include "output.hpp"
#include "preprocessing.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::string inputPath = argv[1];
        const double minSupport = parseRatio(argv[2], "min_support");
        const double minConfidence = parseRatio(argv[3], "min_confidence");

        std::vector<RawLabeledTransaction> rawTransactions = loadRawLabeledTransactions(inputPath);
        std::vector<LabeledTransaction> labeledTransactions = binLabeledTransactions(rawTransactions);
        Transactions featureTransactions = getFeatureTransactions(labeledTransactions);
        const int minSupportCount = static_cast<int>(std::ceil(minSupport * featureTransactions.size()));

        std::map<Itemset, int> frequentItemsets = mineFrequentItemsets(
            featureTransactions,
            minSupportCount
        );

        std::vector<ClassRule> rules = generateClassificationRules(
            frequentItemsets,
            labeledTransactions,
            minConfidence
        );

        printClassificationRules(
            rules,
            static_cast<int>(labeledTransactions.size())
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
