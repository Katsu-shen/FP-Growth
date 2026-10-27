#include "classifier.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t error, const char* action) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + action + ": " + cudaGetErrorString(error));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : pointer_(nullptr) {
        if (bytes != 0) {
            checkCuda(cudaMalloc(&pointer_, bytes), "allocation failed");
        }
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const {
        return pointer_;
    }

private:
    void* pointer_;
};

template <typename T>
void copyToDevice(DeviceBuffer& destination, const std::vector<T>& source) {
    if (!source.empty()) {
        checkCuda(
            cudaMemcpy(destination.get(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
            "host-to-device copy failed"
        );
    }
}

template <typename T>
void copyToHost(std::vector<T>& destination, const DeviceBuffer& source) {
    if (!destination.empty()) {
        checkCuda(
            cudaMemcpy(destination.data(), source.get(), destination.size() * sizeof(T), cudaMemcpyDeviceToHost),
            "device-to-host copy failed"
        );
    }
}

__global__ void countClassMatches(
    const int* candidateItems,
    const int* candidateOffsets,
    const int* transactionItems,
    const int* transactionOffsets,
    const int* transactionClasses,
    int transactionCount,
    int classCount,
    int* counts,
    std::size_t workItems
) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= workItems) {
        return;
    }

    int candidate = static_cast<int>(index / transactionCount);
    int transaction = static_cast<int>(index % transactionCount);
    int candidateIndex = candidateOffsets[candidate];
    const int candidateEnd = candidateOffsets[candidate + 1];
    int transactionIndex = transactionOffsets[transaction];
    const int transactionEnd = transactionOffsets[transaction + 1];

    while (candidateIndex < candidateEnd && transactionIndex < transactionEnd) {
        int candidateItem = candidateItems[candidateIndex];
        int transactionItem = transactionItems[transactionIndex];
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
        atomicAdd(&counts[candidate * classCount + transactionClasses[transaction]], 1);
    }
}

template <typename T>
std::size_t byteCount(const std::vector<T>& values) {
    if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("CUDA input is too large to allocate.");
    }
    return values.size() * sizeof(T);
}

} // namespace

std::vector<ClassRule> generateClassificationRulesCuda(
    const std::map<Itemset, int>& frequentItemsets,
    const std::vector<LabeledTransaction>& transactions,
    int minSupportCount,
    double minConfidence,
    const std::string& targetClass
) {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery failed");
    if (deviceCount == 0) {
        throw std::runtime_error("CUDA backend requested, but no NVIDIA CUDA device is available.");
    }

    if (frequentItemsets.empty() || transactions.empty()) {
        return {};
    }

    std::map<std::string, int> itemIds;
    std::map<std::string, int> classIds;
    std::vector<std::string> classLabels;
    for (const LabeledTransaction& transaction : transactions) {
        for (const std::string& item : transaction.items) {
            itemIds.emplace(item, 0);
        }
        if (classIds.emplace(transaction.classLabel, static_cast<int>(classIds.size())).second) {
            classLabels.push_back(transaction.classLabel);
        }
    }

    if (itemIds.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        classIds.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        transactions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("CUDA input exceeds the supported integer index range.");
    }
    int nextItemId = 0;
    for (auto& entry : itemIds) {
        entry.second = nextItemId++;
    }

    std::vector<Itemset> candidates;
    std::vector<int> candidateSupports;
    std::vector<int> candidateItems;
    std::vector<int> candidateOffsets{0};
    for (const auto& entry : frequentItemsets) {
        candidates.push_back(entry.first);
        candidateSupports.push_back(entry.second);
        for (const std::string& item : entry.first) {
            if (candidateItems.size() == static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("CUDA candidate items exceed the supported index range.");
            }
            candidateItems.push_back(itemIds.at(item));
        }
        candidateOffsets.push_back(static_cast<int>(candidateItems.size()));
    }

    std::vector<int> transactionItems;
    std::vector<int> transactionOffsets{0};
    std::vector<int> transactionClasses;
    for (const LabeledTransaction& transaction : transactions) {
        for (const std::string& item : transaction.items) {
            if (transactionItems.size() == static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("CUDA transaction items exceed the supported index range.");
            }
            transactionItems.push_back(itemIds.at(item));
        }
        transactionOffsets.push_back(static_cast<int>(transactionItems.size()));
        transactionClasses.push_back(classIds.at(transaction.classLabel));
    }

    const std::size_t candidateCount = candidates.size();
    const std::size_t classCount = classLabels.size();
    const std::size_t transactionCount = transactions.size();
    if (candidateCount > std::numeric_limits<std::size_t>::max() / classCount ||
        candidateCount > std::numeric_limits<std::size_t>::max() / transactionCount) {
        throw std::runtime_error("CUDA rule-counting grid is too large.");
    }

    const std::size_t countSize = candidateCount * classCount;
    const std::size_t workItems = candidateCount * transactionCount;
    std::vector<int> hostCounts(countSize, 0);
    DeviceBuffer deviceCandidateItems(byteCount(candidateItems));
    DeviceBuffer deviceCandidateOffsets(byteCount(candidateOffsets));
    DeviceBuffer deviceTransactionItems(byteCount(transactionItems));
    DeviceBuffer deviceTransactionOffsets(byteCount(transactionOffsets));
    DeviceBuffer deviceTransactionClasses(byteCount(transactionClasses));
    DeviceBuffer deviceCounts(byteCount(hostCounts));
    copyToDevice(deviceCandidateItems, candidateItems);
    copyToDevice(deviceCandidateOffsets, candidateOffsets);
    copyToDevice(deviceTransactionItems, transactionItems);
    copyToDevice(deviceTransactionOffsets, transactionOffsets);
    copyToDevice(deviceTransactionClasses, transactionClasses);
    checkCuda(cudaMemset(deviceCounts.get(), 0, byteCount(hostCounts)), "count initialization failed");

    constexpr int threadsPerBlock = 256;
    const std::size_t blocks =
        workItems / threadsPerBlock + (workItems % threadsPerBlock == 0 ? 0 : 1);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("CUDA rule-counting grid exceeds the supported size.");
    }
    countClassMatches<<<static_cast<unsigned int>(blocks), threadsPerBlock>>>(
        static_cast<const int*>(deviceCandidateItems.get()),
        static_cast<const int*>(deviceCandidateOffsets.get()),
        static_cast<const int*>(deviceTransactionItems.get()),
        static_cast<const int*>(deviceTransactionOffsets.get()),
        static_cast<const int*>(deviceTransactionClasses.get()),
        static_cast<int>(transactionCount),
        static_cast<int>(classCount),
        static_cast<int*>(deviceCounts.get()),
        workItems
    );
    checkCuda(cudaGetLastError(), "kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "kernel execution failed");
    copyToHost(hostCounts, deviceCounts);

    std::vector<ClassRule> rules;
    for (std::size_t candidate = 0; candidate < candidateCount; ++candidate) {
        for (std::size_t classIndex = 0; classIndex < classCount; ++classIndex) {
            const int count = hostCounts[candidate * classCount + classIndex];
            if (count < minSupportCount ||
                (!targetClass.empty() && classLabels[classIndex] != targetClass)) {
                continue;
            }

            const double support = static_cast<double>(count) / transactions.size();
            const double confidence = static_cast<double>(count) / candidateSupports[candidate];
            if (confidence >= minConfidence) {
                rules.push_back({candidates[candidate], classLabels[classIndex], count, support, confidence});
            }
        }
    }

    sortClassificationRules(rules);
    return rules;
}
