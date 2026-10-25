#ifndef FP_TREE_HPP
#define FP_TREE_HPP

#include <map>

#include "common.hpp"

std::map<std::string, int> countItemSupport(const Transactions& transactions);
Itemset filterAndSortTransaction(
    const Itemset& transaction,
    const std::map<std::string, int>& supportCounts,
    int minSupportCount
);
Transactions filterAndSortTransactions(
    const Transactions& transactions,
    const std::map<std::string, int>& supportCounts,
    int minSupportCount
);
Itemset canonicalItemset(Itemset itemset);
std::map<Itemset, int> mineFrequentItemsets(
    const Transactions& transactions,
    int minSupportCount,
    std::size_t maxItemsetLength
);

#endif
