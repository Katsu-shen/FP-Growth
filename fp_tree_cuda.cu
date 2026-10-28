#include "fp_tree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ItemId = std::uint32_t;
using IdItemset = std::vector<ItemId>;
using IdTransactions = std::vector<IdItemset>;

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : pointer_(nullptr) {
        if (bytes != 0) {
            check(cudaMalloc(&pointer_, bytes), "allocation failed");
        }
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return pointer_; }

    static void check(cudaError_t error, const char* action) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA ") + action + ": " + cudaGetErrorString(error));
        }
    }

private:
    void* pointer_;
};

template <typename T>
std::size_t byteCount(const std::vector<T>& values) {
    if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("CUDA mining input is too large to allocate.");
    }
    return values.size() * sizeof(T);
}

template <typename T>
void copyToDevice(DeviceBuffer& destination, const std::vector<T>& source) {
    if (!source.empty()) {
        DeviceBuffer::check(
            cudaMemcpy(destination.get(), source.data(), byteCount(source), cudaMemcpyHostToDevice),
            "copy to device failed"
        );
    }
}

template <typename T>
void copyToHost(std::vector<T>& destination, const DeviceBuffer& source) {
    if (!destination.empty()) {
        DeviceBuffer::check(
            cudaMemcpy(destination.data(), source.get(), byteCount(destination), cudaMemcpyDeviceToHost),
            "copy to host failed"
        );
    }
}

__global__ void countCandidateSupports(
    const std::uint32_t* candidateItems,
    const int* candidateOffsets,
    const std::uint32_t* transactionItems,
    const int* transactionOffsets,
    int transactionCount,
    int* supports,
    std::size_t workItems
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= workItems) {
        return;
    }

    const int candidate = static_cast<int>(index / transactionCount);
    const int transaction = static_cast<int>(index % transactionCount);
    int candidateIndex = candidateOffsets[candidate];
    const int candidateEnd = candidateOffsets[candidate + 1];
    int transactionIndex = transactionOffsets[transaction];
    const int transactionEnd = transactionOffsets[transaction + 1];

    while (candidateIndex < candidateEnd && transactionIndex < transactionEnd) {
        const std::uint32_t candidateItem = candidateItems[candidateIndex];
        const std::uint32_t transactionItem = transactionItems[transactionIndex];
        if (candidateItem == transactionItem) {
            ++candidateIndex;
            ++transactionIndex;
        } else if (candidateItem > transactionItem) {
            ++transactionIndex;
        } else {
            return;
        }
    }

    if (candidateIndex == candidateEnd) {
        atomicAdd(&supports[candidate], 1);
    }
}

struct Dictionary {
    std::map<std::string, ItemId> toId;
    std::vector<std::string> toString;
};

Dictionary buildDictionary(const Transactions& transactions) {
    Dictionary dictionary;
    for (const Itemset& transaction : transactions) {
        for (const std::string& item : transaction) {
            dictionary.toId.emplace(item, 0);
        }
    }
    if (dictionary.toId.size() > std::numeric_limits<ItemId>::max()) {
        throw std::runtime_error("CUDA mining has too many distinct items.");
    }
    dictionary.toString.reserve(dictionary.toId.size());
    ItemId nextId = 0;
    for (auto& entry : dictionary.toId) {
        entry.second = nextId++;
        dictionary.toString.push_back(entry.first);
    }
    return dictionary;
}

IdTransactions encodeTransactions(const Transactions& transactions, const Dictionary& dictionary) {
    IdTransactions encoded;
    encoded.reserve(transactions.size());
    for (const Itemset& transaction : transactions) {
        IdItemset ids;
        ids.reserve(transaction.size());
        for (const std::string& item : transaction) {
            ids.push_back(dictionary.toId.at(item));
        }
        encoded.push_back(std::move(ids));
    }
    return encoded;
}

struct DeviceTransactions {
    std::vector<ItemId> items;
    std::vector<int> offsets;
    std::unique_ptr<DeviceBuffer> deviceItems;
    std::unique_ptr<DeviceBuffer> deviceOffsets;

    explicit DeviceTransactions(const IdTransactions& transactions)
        : offsets(1, 0) {
        if (transactions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("CUDA mining has too many transactions.");
        }
        for (const IdItemset& transaction : transactions) {
            if (items.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - transaction.size()) {
                throw std::runtime_error("CUDA mining transaction items exceed the supported index range.");
            }
            items.insert(items.end(), transaction.begin(), transaction.end());
            offsets.push_back(static_cast<int>(items.size()));
        }
        deviceItems = std::make_unique<DeviceBuffer>(byteCount(items));
        deviceOffsets = std::make_unique<DeviceBuffer>(byteCount(offsets));
        copyToDevice(*deviceItems, items);
        copyToDevice(*deviceOffsets, offsets);
    }
};

std::vector<int> countSupportsCuda(
    const std::vector<IdItemset>& candidates,
    const DeviceTransactions& transactions
) {
    constexpr std::size_t maxCandidatesPerBatch = 4096;
    constexpr int threadsPerBlock = 256;
    const std::size_t transactionCount = transactions.offsets.size() - 1;
    std::vector<int> supports(candidates.size(), 0);

    for (std::size_t begin = 0; begin < candidates.size(); begin += maxCandidatesPerBatch) {
        const std::size_t end = std::min(candidates.size(), begin + maxCandidatesPerBatch);
        std::vector<ItemId> candidateItems;
        std::vector<int> candidateOffsets{0};
        for (std::size_t index = begin; index < end; ++index) {
            const IdItemset& candidate = candidates[index];
            if (candidateItems.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - candidate.size()) {
                throw std::runtime_error("CUDA mining candidate items exceed the supported index range.");
            }
            candidateItems.insert(candidateItems.end(), candidate.begin(), candidate.end());
            candidateOffsets.push_back(static_cast<int>(candidateItems.size()));
        }

        const std::size_t candidateCount = end - begin;
        if (candidateCount > std::numeric_limits<std::size_t>::max() / transactionCount) {
            throw std::runtime_error("CUDA mining grid is too large.");
        }
        const std::size_t workItems = candidateCount * transactionCount;
        const std::size_t blocks = (workItems + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error("CUDA mining grid exceeds the supported size.");
        }

        std::vector<int> batchSupports(candidateCount, 0);
        DeviceBuffer deviceCandidateItems(byteCount(candidateItems));
        DeviceBuffer deviceCandidateOffsets(byteCount(candidateOffsets));
        DeviceBuffer deviceSupports(byteCount(batchSupports));
        copyToDevice(deviceCandidateItems, candidateItems);
        copyToDevice(deviceCandidateOffsets, candidateOffsets);
        DeviceBuffer::check(cudaMemset(deviceSupports.get(), 0, byteCount(batchSupports)), "count initialization failed");
        countCandidateSupports<<<static_cast<unsigned int>(blocks), threadsPerBlock>>>(
            static_cast<const ItemId*>(deviceCandidateItems.get()),
            static_cast<const int*>(deviceCandidateOffsets.get()),
            static_cast<const ItemId*>(transactions.deviceItems->get()),
            static_cast<const int*>(transactions.deviceOffsets->get()),
            static_cast<int>(transactionCount),
            static_cast<int*>(deviceSupports.get()),
            workItems
        );
        DeviceBuffer::check(cudaGetLastError(), "mining kernel launch failed");
        DeviceBuffer::check(cudaDeviceSynchronize(), "mining kernel execution failed");
        copyToHost(batchSupports, deviceSupports);
        std::copy(batchSupports.begin(), batchSupports.end(), supports.begin() + begin);
    }

    return supports;
}

bool hasFrequentSubsets(const IdItemset& candidate, const std::set<IdItemset>& previousLevel) {
    for (std::size_t omitted = 0; omitted < candidate.size(); ++omitted) {
        IdItemset subset;
        subset.reserve(candidate.size() - 1);
        for (std::size_t index = 0; index < candidate.size(); ++index) {
            if (index != omitted) {
                subset.push_back(candidate[index]);
            }
        }
        if (previousLevel.find(subset) == previousLevel.end()) {
            return false;
        }
    }
    return true;
}

std::vector<IdItemset> generateNextLevel(const std::vector<IdItemset>& previousLevel) {
    std::vector<IdItemset> candidates;
    if (previousLevel.empty()) {
        return candidates;
    }
    const std::size_t prefixLength = previousLevel.front().size() - 1;
    const std::set<IdItemset> previousSet(previousLevel.begin(), previousLevel.end());

    for (std::size_t left = 0; left < previousLevel.size(); ++left) {
        for (std::size_t right = left + 1; right < previousLevel.size(); ++right) {
            if (!std::equal(
                    previousLevel[left].begin(), previousLevel[left].begin() + prefixLength,
                    previousLevel[right].begin()
                )) {
                break;
            }
            IdItemset candidate = previousLevel[left];
            candidate.push_back(previousLevel[right].back());
            if (hasFrequentSubsets(candidate, previousSet)) {
                candidates.push_back(std::move(candidate));
            }
        }
    }
    return candidates;
}

Itemset decode(const IdItemset& ids, const Dictionary& dictionary) {
    Itemset items;
    items.reserve(ids.size());
    for (ItemId id : ids) {
        items.push_back(dictionary.toString.at(id));
    }
    return items;
}

} // namespace

std::map<Itemset, int> mineFrequentItemsetsCuda(
    const Transactions& transactions,
    int minSupportCount,
    std::size_t maxItemsetLength
) {
    int deviceCount = 0;
    DeviceBuffer::check(cudaGetDeviceCount(&deviceCount), "device discovery failed");
    if (deviceCount == 0) {
        throw std::runtime_error("CUDA mining requested, but no NVIDIA CUDA device is available.");
    }
    if (transactions.empty()) {
        return {};
    }

    const Dictionary dictionary = buildDictionary(transactions);
    const IdTransactions encoded = encodeTransactions(transactions, dictionary);
    const DeviceTransactions deviceTransactions(encoded);
    std::vector<IdItemset> candidates;
    candidates.reserve(dictionary.toString.size());
    for (std::size_t index = 0; index < dictionary.toString.size(); ++index) {
        candidates.push_back({static_cast<ItemId>(index)});
    }

    std::map<Itemset, int> frequentItemsets;
    for (std::size_t length = 1; !candidates.empty() && (maxItemsetLength == 0 || length <= maxItemsetLength); ++length) {
        const std::vector<int> supports = countSupportsCuda(candidates, deviceTransactions);
        std::vector<IdItemset> frequentLevel;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (supports[index] >= minSupportCount) {
                frequentItemsets.emplace(decode(candidates[index], dictionary), supports[index]);
                frequentLevel.push_back(std::move(candidates[index]));
            }
        }
        if (frequentLevel.empty() || (maxItemsetLength != 0 && length == maxItemsetLength)) {
            break;
        }
        candidates = generateNextLevel(frequentLevel);
    }
    return frequentItemsets;
}
