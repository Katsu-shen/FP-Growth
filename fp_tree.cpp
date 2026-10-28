#include "fp_tree.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// These public helpers keep their original string-facing API.  The mining path
// below converts each item once and uses integer IDs thereafter.
std::map<std::string, int> countItemSupport(const Transactions& transactions) {
    std::map<std::string, int> counts;
    for (const auto& transaction : transactions)
        for (const auto& item : transaction) ++counts[item];
    return counts;
}

Itemset filterAndSortTransaction(const Itemset& transaction,
                                 const std::map<std::string, int>& counts,
                                 int minSupportCount) {
    Itemset filtered;
    for (const auto& item : transaction) {
        auto found = counts.find(item);
        if (found != counts.end() && found->second >= minSupportCount) filtered.push_back(item);
    }
    std::sort(filtered.begin(), filtered.end(), [&](const std::string& left, const std::string& right) {
        return counts.at(left) != counts.at(right) ? counts.at(left) > counts.at(right) : left < right;
    });
    return filtered;
}

Transactions filterAndSortTransactions(const Transactions& transactions,
                                       const std::map<std::string, int>& counts,
                                       int minSupportCount) {
    Transactions result;
    for (const auto& transaction : transactions) {
        Itemset filtered = filterAndSortTransaction(transaction, counts, minSupportCount);
        if (!filtered.empty()) result.push_back(std::move(filtered));
    }
    return result;
}

Itemset canonicalItemset(Itemset itemset) {
    std::sort(itemset.begin(), itemset.end());
    return itemset;
}

namespace {
using ItemId = std::uint32_t;
using IdItemset = std::vector<ItemId>;
using IdTransactions = std::vector<IdItemset>;

struct WeightedIdTransaction { IdItemset items; int count; };
struct Dictionary { std::map<std::string, ItemId> toId; std::vector<std::string> toString; };
struct FPNode {
    ItemId item;
    int count;
    FPNode* parent;
    std::map<ItemId, std::unique_ptr<FPNode>> children;
    FPNode* next;
};
struct HeaderEntry { int support; FPNode* head; FPNode* tail; };
struct FPTree { std::unique_ptr<FPNode> root; std::map<ItemId, HeaderEntry> headerTable; };

static Dictionary buildDictionary(const Transactions& transactions) {
    Dictionary dictionary;
    for (const auto& transaction : transactions)
        for (const auto& item : transaction) dictionary.toId.emplace(item, 0);
    dictionary.toString.reserve(dictionary.toId.size());
    ItemId id = 0;
    // Lexical ID order preserves the original lexical tie-breaker.
    for (auto& entry : dictionary.toId) { entry.second = id++; dictionary.toString.push_back(entry.first); }
    return dictionary;
}

static IdTransactions encodeTransactions(const Transactions& transactions, const Dictionary& dictionary) {
    IdTransactions encoded;
    encoded.reserve(transactions.size());
    for (const auto& transaction : transactions) {
        IdItemset ids;
        ids.reserve(transaction.size());
        for (const auto& item : transaction) ids.push_back(dictionary.toId.at(item));
        encoded.push_back(std::move(ids));
    }
    return encoded;
}

static std::map<ItemId, int> countIds(const IdTransactions& transactions) {
    std::map<ItemId, int> counts;
    for (const auto& transaction : transactions)
        for (ItemId item : transaction) ++counts[item];
    return counts;
}

static IdItemset filterAndSortIds(const IdItemset& transaction,
                                  const std::map<ItemId, int>& counts,
                                  int minSupportCount) {
    IdItemset filtered;
    for (ItemId item : transaction) {
        auto found = counts.find(item);
        if (found != counts.end() && found->second >= minSupportCount) filtered.push_back(item);
    }
    std::sort(filtered.begin(), filtered.end(), [&](ItemId left, ItemId right) {
        return counts.at(left) != counts.at(right) ? counts.at(left) > counts.at(right) : left < right;
    });
    return filtered;
}

static IdTransactions filterAndSortIds(const IdTransactions& transactions,
                                       const std::map<ItemId, int>& counts,
                                       int minSupportCount) {
    IdTransactions result;
    result.reserve(transactions.size());
    for (const auto& transaction : transactions) {
        IdItemset filtered = filterAndSortIds(transaction, counts, minSupportCount);
        if (!filtered.empty()) result.push_back(std::move(filtered));
    }
    return result;
}

static std::unique_ptr<FPNode> createNode(ItemId item, int count, FPNode* parent) {
    return std::make_unique<FPNode>(FPNode{item, count, parent, {}, nullptr});
}

static void linkHeaderNode(HeaderEntry& entry, FPNode* node) {
    if (entry.head == nullptr) entry.head = entry.tail = node;
    else { entry.tail->next = node; entry.tail = node; }
}

static void insertTransaction(FPTree& tree, const IdItemset& transaction, int count) {
    FPNode* current = tree.root.get();
    for (ItemId item : transaction) {
        auto child = current->children.find(item);
        if (child == current->children.end()) {
            auto node = createNode(item, count, current);
            FPNode* raw = node.get();
            current->children.emplace(item, std::move(node));
            linkHeaderNode(tree.headerTable.at(item), raw);
            current = raw;
        } else { child->second->count += count; current = child->second.get(); }
    }
}

static FPTree buildTree(const std::vector<WeightedIdTransaction>& transactions,
                        const std::map<ItemId, int>& counts) {
    FPTree tree;
    tree.root = createNode(0, 0, nullptr);
    for (const auto& count : counts) tree.headerTable.emplace(count.first, HeaderEntry{count.second, nullptr, nullptr});
    for (const auto& transaction : transactions) insertTransaction(tree, transaction.items, transaction.count);
    return tree;
}

static FPTree buildTree(const IdTransactions& transactions, const std::map<ItemId, int>& counts) {
    std::vector<WeightedIdTransaction> weighted;
    weighted.reserve(transactions.size());
    for (const auto& transaction : transactions) weighted.push_back({transaction, 1});
    return buildTree(weighted, counts);
}

static std::vector<ItemId> itemsByAscendingSupport(const std::map<ItemId, HeaderEntry>& headers) {
    std::vector<ItemId> items;
    for (const auto& entry : headers)
        if (entry.second.support > 0 && entry.second.head != nullptr) items.push_back(entry.first);
    std::sort(items.begin(), items.end(), [&](ItemId left, ItemId right) {
        return headers.at(left).support != headers.at(right).support
            ? headers.at(left).support < headers.at(right).support : left < right;
    });
    return items;
}

static std::vector<WeightedIdTransaction> conditionalBase(const FPTree& tree, ItemId item) {
    std::vector<WeightedIdTransaction> base;
    auto header = tree.headerTable.find(item);
    if (header == tree.headerTable.end()) return base;
    for (FPNode* node = header->second.head; node != nullptr; node = node->next) {
        IdItemset path;
        for (FPNode* parent = node->parent; parent != tree.root.get(); parent = parent->parent) path.push_back(parent->item);
        if (!path.empty()) { std::reverse(path.begin(), path.end()); base.push_back({std::move(path), node->count}); }
    }
    return base;
}

static std::map<ItemId, int> countWeighted(const std::vector<WeightedIdTransaction>& transactions) {
    std::map<ItemId, int> counts;
    for (const auto& transaction : transactions)
        for (ItemId item : transaction.items) counts[item] += transaction.count;
    return counts;
}

static FPTree conditionalTree(const std::vector<WeightedIdTransaction>& base, int minSupportCount) {
    std::map<ItemId, int> counts = countWeighted(base);
    for (auto iterator = counts.begin(); iterator != counts.end();)
        if (iterator->second < minSupportCount) iterator = counts.erase(iterator); else ++iterator;
    std::vector<WeightedIdTransaction> filtered;
    filtered.reserve(base.size());
    for (const auto& transaction : base) {
        IdItemset items = filterAndSortIds(transaction.items, counts, minSupportCount);
        if (!items.empty()) filtered.push_back({std::move(items), transaction.count});
    }
    return buildTree(filtered, counts);
}

static void mineTree(const FPTree& tree, int minSupportCount, std::size_t maxItemsetLength,
                     const IdItemset& suffix, std::map<IdItemset, int>& itemsets) {
    for (ItemId item : itemsByAscendingSupport(tree.headerTable)) {
        const HeaderEntry& header = tree.headerTable.at(item);
        if (header.support < minSupportCount) continue;
        IdItemset pattern = suffix;
        pattern.push_back(item);
        std::sort(pattern.begin(), pattern.end());
        itemsets[pattern] = header.support;
        FPTree conditional = conditionalTree(conditionalBase(tree, item), minSupportCount);
        if ((maxItemsetLength == 0 || pattern.size() < maxItemsetLength) && !conditional.headerTable.empty())
            mineTree(conditional, minSupportCount, maxItemsetLength, pattern, itemsets);
    }
}
}  // namespace

std::map<Itemset, int> mineFrequentItemsets(const Transactions& transactions, int minSupportCount,
                                            std::size_t maxItemsetLength) {
    Dictionary dictionary = buildDictionary(transactions);
    IdTransactions encoded = encodeTransactions(transactions, dictionary);
    std::map<ItemId, int> counts = countIds(encoded);
    IdTransactions ordered = filterAndSortIds(encoded, counts, minSupportCount);
    for (auto iterator = counts.begin(); iterator != counts.end();)
        if (iterator->second < minSupportCount) iterator = counts.erase(iterator); else ++iterator;
    if (ordered.empty() || counts.empty()) return {};
    FPTree tree = buildTree(ordered, counts);
    std::map<IdItemset, int> encodedItemsets;
    mineTree(tree, minSupportCount, maxItemsetLength, {}, encodedItemsets);
    std::map<Itemset, int> decoded;
    for (const auto& entry : encodedItemsets) {
        Itemset itemset;
        for (ItemId item : entry.first) itemset.push_back(dictionary.toString.at(item));
        decoded.emplace(std::move(itemset), entry.second);
    }
    return decoded;
}
