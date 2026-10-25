#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "classifier.hpp"
#include "common.hpp"
#include "data_loader.hpp"
#include "fp_tree.hpp"
#include "output.hpp"
#include "preprocessing.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::string inputPath = argv[1];
        const double minSupport = parseRatio(argv[2], "min_support");
        const double minConfidence = parseRatio(argv[3], "min_confidence");
        std::string targetClass;
        std::size_t maxItemsetLength = 0;
        std::size_t maxRules = 0;

        for (int index = 4; index < argc; index += 2) {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + std::string(argv[index]));
            }

            const std::string option = argv[index];
            const std::string value = argv[index + 1];

            if (option == "--class") {
                targetClass = value;
            } else if (option == "--max-length") {
                maxItemsetLength = parsePositiveCount(value, "max-length");
            } else if (option == "--top") {
                maxRules = parsePositiveCount(value, "top");
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }

        std::vector<RawLabeledTransaction> rawTransactions = loadRawLabeledTransactions(inputPath);
        std::vector<LabeledTransaction> labeledTransactions = binLabeledTransactions(rawTransactions);
        Transactions featureTransactions = getFeatureTransactions(labeledTransactions);
        const int minSupportCount = static_cast<int>(std::ceil(minSupport * featureTransactions.size()));

        std::map<Itemset, int> frequentItemsets = mineFrequentItemsets(
            featureTransactions,
            minSupportCount,
            maxItemsetLength
        );

        std::vector<ClassRule> rules = generateClassificationRules(
            frequentItemsets,
            labeledTransactions,
            minConfidence,
            targetClass
        );

        printClassificationRules(
            rules,
            static_cast<int>(labeledTransactions.size()),
            maxRules
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
