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
        bool hasHeader = true;
#ifdef FP_GROWTH_CUDA
        std::string backend = "cuda";
        std::string miningBackend = "cuda";
#else
        std::string backend = "cpu";
        std::string miningBackend = "cpu";
#endif

        for (int index = 4; index < argc;) {
            const std::string option = argv[index];
            if (option == "--header") {
                hasHeader = true;
                ++index;
                continue;
            }
            if (option == "--no-header") {
                hasHeader = false;
                ++index;
                continue;
            }
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + std::string(argv[index]));
            }

            const std::string value = argv[index + 1];

            if (option == "--class") {
                targetClass = value;
            } else if (option == "--max-length") {
                maxItemsetLength = parsePositiveCount(value, "max-length");
            } else if (option == "--top") {
                maxRules = parsePositiveCount(value, "top");
            } else if (option == "--backend") {
                if (value != "cpu" && value != "cuda") {
                    throw std::runtime_error("backend must be cpu or cuda.");
                }
                backend = value;
            } else if (option == "--mining-backend") {
                if (value != "cpu" && value != "cuda") {
                    throw std::runtime_error("mining-backend must be cpu or cuda.");
                }
                miningBackend = value;
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
            index += 2;
        }

        LoadedLabeledTransactions loaded = loadRawLabeledTransactions(inputPath, hasHeader);
        std::vector<LabeledTransaction> labeledTransactions = binLabeledTransactions(
            loaded.transactions, loaded.featureNames
        );
        Transactions featureTransactions = getFeatureTransactions(labeledTransactions);
        const int minSupportCount = static_cast<int>(std::ceil(minSupport * featureTransactions.size()));

        std::map<Itemset, int> frequentItemsets;
#ifdef FP_GROWTH_CUDA
        if (miningBackend == "cuda") {
            frequentItemsets = mineFrequentItemsetsCuda(
                featureTransactions, minSupportCount, maxItemsetLength
            );
        } else {
            frequentItemsets = mineFrequentItemsets(
                featureTransactions, minSupportCount, maxItemsetLength
            );
        }
#else
        if (miningBackend == "cuda") {
            throw std::runtime_error(
                "CUDA mining support is not included in this executable. Build and run fp_growth_cuda."
            );
        }
        frequentItemsets = mineFrequentItemsets(
            featureTransactions, minSupportCount, maxItemsetLength
        );
#endif

        std::vector<ClassRule> rules;
#ifdef FP_GROWTH_CUDA
        if (backend == "cuda") {
            rules = generateClassificationRulesCuda(
                frequentItemsets, labeledTransactions, minSupportCount, minConfidence, targetClass
            );
        } else {
            rules = generateClassificationRules(
                frequentItemsets, labeledTransactions, minSupportCount, minConfidence, targetClass
            );
        }
#else
        if (backend == "cuda") {
            throw std::runtime_error(
                "CUDA support is not included in this executable. Build and run fp_growth_cuda."
            );
        }
        rules = generateClassificationRules(
            frequentItemsets, labeledTransactions, minSupportCount, minConfidence, targetClass
        );
#endif

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
