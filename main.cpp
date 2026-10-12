#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Itemset = std::vector<std::string>;
using Transactions = std::vector<Itemset>;

// nodes of the fp-tree
// possible comparison of this with a linked list instead of map of children
struct FPNode {
    std::string item;
    int count;
    FPNode* parent;
    std::map<std::string, std::unique_ptr<FPNode>> children; // map of the children under this node
    FPNode* next;
};

// header table
// keeps support count and links to each node with that item
struct HeaderEntry {
    int support;
    FPNode* head;
    FPNode* tail;
};

// stores the root node and the header table together
struct FPTree {
    std::unique_ptr<FPNode> root;
    std::map<std::string, HeaderEntry> headerTable;
};

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

// clean data before we start mining data
static std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(start, end - start);
}

// split one csv row by commas
static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> values;
    std::string token;
    std::stringstream stream(line);

    while (std::getline(stream, token, ',')) {
        values.push_back(trim(token));
    }

    return values;
}

// check if a value can be parsed as a number
static bool parseDouble(const std::string& value, double& parsed) {
    std::stringstream stream(value);
    stream >> parsed;
    stream >> std::ws;
    return stream.eof() && !value.empty();
}

// load csv rows where the last value is the class label
static std::vector<RawLabeledTransaction> loadRawLabeledTransactions(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    std::vector<RawLabeledTransaction> transactions;
    std::size_t expectedFeatureCount = 0;
    std::string line;
    int lineNumber = 0;

    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::vector<std::string> values = splitCsvLine(line);
        if (values.size() < 2) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " needs at least one feature and one class label.");
        }

        std::string classLabel = values.back();
        values.pop_back();

        if (classLabel.empty()) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has an empty class label.");
        }

        if (expectedFeatureCount == 0) {
            expectedFeatureCount = values.size();
        } else if (values.size() != expectedFeatureCount) {
            throw std::runtime_error("Line " + std::to_string(lineNumber) + " has a different number of features.");
        }

        transactions.push_back({values, classLabel});
    }

    if (transactions.empty()) {
        throw std::runtime_error("Input file has no transactions.");
    }

    return transactions;
}

// decide whether every value in each feature column is numeric
static std::vector<bool> findNumericColumns(const std::vector<RawLabeledTransaction>& transactions) {
    std::vector<bool> numericColumns(transactions.front().values.size(), true);

    for (const RawLabeledTransaction& transaction : transactions) {
        for (std::size_t index = 0; index < transaction.values.size(); ++index) {
            double parsed = 0.0;
            if (!parseDouble(transaction.values[index], parsed)) {
                numericColumns[index] = false;
            }
        }
    }

    return numericColumns;
}

// get the min and max value for every numeric column
static void findNumericRanges(
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

            double parsed = 0.0;
            parseDouble(transaction.values[index], parsed);
            mins[index] = std::min(mins[index], parsed);
            maxes[index] = std::max(maxes[index], parsed);
        }
    }
}

// turn a numeric value into low, medium, or high
static std::string binNumericValue(double value, double minValue, double maxValue) {
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
static std::vector<LabeledTransaction> binLabeledTransactions(
    const std::vector<RawLabeledTransaction>& rawTransactions
) {
    std::vector<bool> numericColumns = findNumericColumns(rawTransactions);
    std::vector<double> mins;
    std::vector<double> maxes;
    findNumericRanges(rawTransactions, numericColumns, mins, maxes);

    std::vector<LabeledTransaction> transactions;

    for (const RawLabeledTransaction& rawTransaction : rawTransactions) {
        std::set<std::string> uniqueItems;

        for (std::size_t index = 0; index < rawTransaction.values.size(); ++index) {
            std::string itemName = "attr" + std::to_string(index + 1) + "=";

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
static Transactions getFeatureTransactions(const std::vector<LabeledTransaction>& labeledTransactions) {
    Transactions transactions;

    for (const LabeledTransaction& transaction : labeledTransactions) {
        transactions.push_back(transaction.items);
    }

    return transactions;
}

// count frequency of all items within transaction
static std::map<std::string, int> countItemSupport(const Transactions& transactions) {
    std::map<std::string, int> supportCounts;

    for (const Itemset& transaction : transactions) {
        for (const std::string& item : transaction) {
            ++supportCounts[item];
        }
    }

    return supportCounts;
}

// sort each itemset by the frequency of items in the entire transaction
static Itemset filterAndSortTransaction(
    const Itemset& transaction,
    const std::map<std::string, int>& supportCounts,
    int minSupportCount
) {
    Itemset filtered;

    // add transactions to filtered array if there is enough support
    for (const std::string& item : transaction) {
        auto support = supportCounts.find(item);
        if (support != supportCounts.end() && support->second >= minSupportCount) {
            filtered.push_back(item);
        }
    }

    // sort by the strength of support
    std::sort(filtered.begin(), filtered.end(), [&](const std::string& left, const std::string& right) {
        int leftSupport = supportCounts.at(left);
        int rightSupport = supportCounts.at(right);

        if (leftSupport != rightSupport) {
            return leftSupport > rightSupport;
        }

        return left < right;
    });

    return filtered;
}

// store these filtered itemsets into a new filtered transaction
static Transactions filterAndSortTransactions(
    const Transactions& transactions,
    const std::map<std::string, int>& supportCounts,
    int minSupportCount
) {
    Transactions filteredTransactions;

    for (const Itemset& transaction : transactions) {
        Itemset filtered = filterAndSortTransaction(transaction, supportCounts, minSupportCount);
        if (!filtered.empty()) {
            filteredTransactions.push_back(filtered);
        }
    }

    return filteredTransactions;
}

// create ptrs for tree
static std::unique_ptr<FPNode> createNode(
    const std::string& item,
    int count,
    FPNode* parent
) {
    return std::make_unique<FPNode>(FPNode{item, count, parent, {}, nullptr});
}

// add a node to the linked list for this item in the header table
static void linkHeaderNode(HeaderEntry& entry, FPNode* node) {
    if (entry.head == nullptr) {
        entry.head = node;
        entry.tail = node;
        return;
    }

    entry.tail->next = node;
    entry.tail = node;
}

// build one path in the FP-tree from one transaction
static void insertTransaction(
    FPTree& tree,
    const Itemset& transaction,
    int count
) {
    FPNode* current = tree.root.get();

    for (const std::string& item : transaction) {
        // checks if the header already exists in the tree
        auto child = current->children.find(item);

        // if it doesn't then find() returns children.end()
        if (child == current->children.end()) {
            // insert full transaction into tree
            // unique_ptr for dynamic memory alloc
            std::unique_ptr<FPNode> newNode = createNode(item, count, current);
            FPNode* rawNode = newNode.get(); // tmp pointer since we move og node into children map
            current->children[item] = std::move(newNode);

            // add the header as part of the headertable
            linkHeaderNode(tree.headerTable[item], rawNode);
            current = rawNode;
        } else {
            // increment existing node since this prefix already exists
            child->second->count += count;
            current = child->second.get();
        }
    }
}

// build fp tree from transactions that already include their counts
static FPTree buildFPTree(
    const std::vector<WeightedTransaction>& transactions,
    const std::map<std::string, int>& supportCounts
) {
    FPTree tree;
    tree.root = createNode("", 0, nullptr);

    for (const auto& support : supportCounts) {
        tree.headerTable[support.first] = HeaderEntry{support.second, nullptr, nullptr};
    }

    for (const WeightedTransaction& transaction : transactions) {
        insertTransaction(tree, transaction.items, transaction.count);
    }

    return tree;
}

// build fp tree from normal transactions where each transaction has count 1
static FPTree buildFPTree(
    const Transactions& transactions,
    const std::map<std::string, int>& supportCounts
) {
    std::vector<WeightedTransaction> weightedTransactions;

    for (const Itemset& transaction : transactions) {
        weightedTransactions.push_back({transaction, 1});
    }

    return buildFPTree(weightedTransactions, supportCounts);
}

// get items in order from least frequent to most frequent for mining
static std::vector<std::string> itemsByAscendingSupport(
    const std::map<std::string, HeaderEntry>& headerTable
) {
    std::vector<std::string> items;

    for (const auto& entry : headerTable) {
        if (entry.second.support > 0 && entry.second.head != nullptr) {
            items.push_back(entry.first);
        }
    }

    std::sort(items.begin(), items.end(), [&](const std::string& left, const std::string& right) {
        int leftSupport = headerTable.at(left).support;
        int rightSupport = headerTable.at(right).support;

        // lexicological sorting
        if (leftSupport != rightSupport) {
            return leftSupport < rightSupport;
        }

        return left < right;
    });

    return items;
}

// build all prefix paths that lead into a specific item
static std::vector<WeightedTransaction> buildConditionalPatternBase(
    const FPTree& tree,
    const std::string& item
) {
    std::vector<WeightedTransaction> patternBase;
    auto header = tree.headerTable.find(item);

    if (header == tree.headerTable.end()) {
        return patternBase;
    }

    // follow the header linked list to visit every node for this item
    for (FPNode* node = header->second.head; node != nullptr; node = node->next) {
        Itemset prefixPath;

        // walk upward from the item node to collect the path before it
        for (FPNode* parent = node->parent; parent != nullptr && !parent->item.empty(); parent = parent->parent) {
            prefixPath.push_back(parent->item);
        }

        if (!prefixPath.empty()) {
            // reverse so the path goes root to leaf again
            std::reverse(prefixPath.begin(), prefixPath.end());
            patternBase.push_back({prefixPath, node->count});
        }
    }

    return patternBase;
}

// count support for conditional transactions where each path has a weight
static std::map<std::string, int> countWeightedItemSupport(
    const std::vector<WeightedTransaction>& transactions
) {
    std::map<std::string, int> supportCounts;

    for (const WeightedTransaction& transaction : transactions) {
        for (const std::string& item : transaction.items) {
            supportCounts[item] += transaction.count;
        }
    }

    return supportCounts;
}

// filter conditional transactions and sort them by conditional support
static std::vector<WeightedTransaction> filterAndSortWeightedTransactions(
    const std::vector<WeightedTransaction>& transactions,
    const std::map<std::string, int>& supportCounts,
    int minSupportCount
) {
    std::vector<WeightedTransaction> filteredTransactions;

    for (const WeightedTransaction& transaction : transactions) {
        Itemset filtered = filterAndSortTransaction(transaction.items, supportCounts, minSupportCount);
        if (!filtered.empty()) {
            filteredTransactions.push_back({filtered, transaction.count});
        }
    }

    return filteredTransactions;
}

// build the smaller fp-tree for one conditional pattern base
static FPTree buildConditionalFPTree(
    const std::vector<WeightedTransaction>& patternBase,
    int minSupportCount
) {
    std::map<std::string, int> supportCounts = countWeightedItemSupport(patternBase);

    // remove conditional items that do not pass support
    for (auto iterator = supportCounts.begin(); iterator != supportCounts.end();) {
        if (iterator->second < minSupportCount) {
            iterator = supportCounts.erase(iterator);
        } else {
            ++iterator;
        }
    }

    std::vector<WeightedTransaction> filteredTransactions = filterAndSortWeightedTransactions(
        patternBase,
        supportCounts,
        minSupportCount
    );

    return buildFPTree(filteredTransactions, supportCounts);
}

// sort itemset alphabetically so the output map stores duplicates the same way
static Itemset canonicalItemset(Itemset itemset) {
    std::sort(itemset.begin(), itemset.end());
    return itemset;
}

// recursively mine the fp-tree to find frequent patterns
static void mineFPTree(
    const FPTree& tree,
    int minSupportCount,
    const Itemset& suffix,
    std::map<Itemset, int>& frequentItemsets
) {
    std::vector<std::string> items = itemsByAscendingSupport(tree.headerTable);

    for (const std::string& item : items) {
        const HeaderEntry& header = tree.headerTable.at(item);
        if (header.support < minSupportCount) {
            continue;
        }

        Itemset pattern = suffix;
        pattern.push_back(item);
        frequentItemsets[canonicalItemset(pattern)] = header.support;

        // use this item to build and mine its conditional tree
        std::vector<WeightedTransaction> patternBase = buildConditionalPatternBase(tree, item);
        FPTree conditionalTree = buildConditionalFPTree(patternBase, minSupportCount);

        if (!conditionalTree.headerTable.empty()) {
            mineFPTree(conditionalTree, minSupportCount, pattern, frequentItemsets);
        }
    }
}

// wrapper that starts mining with an empty suffix
static std::map<Itemset, int> mineFPTree(
    const FPTree& tree,
    int minSupportCount
) {
    std::map<Itemset, int> frequentItemsets;
    Itemset suffix;
    mineFPTree(tree, minSupportCount, suffix, frequentItemsets);
    return frequentItemsets;
}

// check if a pattern is inside one transaction
static bool isSubset(const Itemset& pattern, const Itemset& transaction) {
    return std::includes(
        transaction.begin(),
        transaction.end(),
        pattern.begin(),
        pattern.end()
    );
}

// build classification rules from frequent itemsets
static std::vector<ClassRule> generateClassificationRules(
    const std::map<Itemset, int>& frequentItemsets,
    const std::vector<LabeledTransaction>& transactions,
    double minConfidence
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

// print one itemset with commas between items
static std::string formatItemset(const Itemset& itemset) {
    std::ostringstream output;
    output << "{";

    for (std::size_t index = 0; index < itemset.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << itemset[index];
    }

    output << "}";
    return output.str();
}

// print class rules with support ratio, confidence, and count
static void printClassificationRules(
    const std::vector<ClassRule>& rules,
    int transactionCount
) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Transactions: " << transactionCount << "\n\n";
    std::cout << "Classification rules\n";

    if (rules.empty()) {
        std::cout << "  No rules met the support and confidence thresholds.\n";
        return;
    }

    for (const ClassRule& rule : rules) {
        std::cout << "  " << std::setw(40) << std::left << formatItemset(rule.antecedent)
                  << " -> {" << rule.classLabel << "}"
                  << " support=" << rule.support
                  << " confidence=" << rule.confidence
                  << " count=" << rule.count << "\n";
    }
}

// parse min support ratio or min confidence from command line
static double parseRatio(const std::string& value, const std::string& name) {
    double parsed = 0.0;
    std::stringstream stream(value);
    stream >> parsed;
    stream >> std::ws;

    if (!stream.eof() || parsed < 0.0 || parsed > 1.0) {
        throw std::runtime_error(name + " must be a number between 0 and 1.");
    }

    return parsed;
}

// explain how to run the program if the arguments are wrong
static void printUsage(const char* programName) {
    std::cerr
        << "Usage: " << programName << " <data.csv> <min_support> <min_confidence>\n"
        << "Example: " << programName << " iris/iris.data 0.2 0.7\n";
}

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

        // first scan counts each item before building the tree
        std::map<std::string, int> supportCounts = countItemSupport(featureTransactions);
        Transactions orderedTransactions = filterAndSortTransactions(
            featureTransactions,
            supportCounts,
            minSupportCount
        );

        // remove items that do not meet min support
        for (auto iterator = supportCounts.begin(); iterator != supportCounts.end();) {
            if (iterator->second < minSupportCount) {
                iterator = supportCounts.erase(iterator);
            } else {
                ++iterator;
            }
        }

        std::map<Itemset, int> frequentItemsets;
        if (!orderedTransactions.empty() && !supportCounts.empty()) {
            // build the tree and mine all patterns from it
            FPTree tree = buildFPTree(orderedTransactions, supportCounts);
            frequentItemsets = mineFPTree(tree, minSupportCount);
        }

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
