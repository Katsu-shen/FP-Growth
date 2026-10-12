#include "fp_tree.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

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

// count frequency of all items within transaction
std::map<std::string, int> countItemSupport(const Transactions& transactions) {
    std::map<std::string, int> supportCounts;

    for (const Itemset& transaction : transactions) {
        for (const std::string& item : transaction) {
            ++supportCounts[item];
        }
    }

    return supportCounts;
}

// sort each itemset by the frequency of items in the entire transaction
Itemset filterAndSortTransaction(
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
Transactions filterAndSortTransactions(
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
Itemset canonicalItemset(Itemset itemset) {
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

// run the full fp-growth flow for feature transactions
std::map<Itemset, int> mineFrequentItemsets(
    const Transactions& transactions,
    int minSupportCount
) {
    std::map<std::string, int> supportCounts = countItemSupport(transactions);
    Transactions orderedTransactions = filterAndSortTransactions(
        transactions,
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

    if (orderedTransactions.empty() || supportCounts.empty()) {
        return {};
    }

    FPTree tree = buildFPTree(orderedTransactions, supportCounts);
    return mineFPTree(tree, minSupportCount);
}
