#pragma once

#include "types.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace pulse {

struct KvPage {
    uint32_t page_id;
    int ref_count;
    uint64_t last_accessed_ns;
    void* device_ptr;
};

class PagedKvPool {
public:
    PagedKvPool(size_t max_pool_bytes = KV_POOL_RESERVED_BYTES)
        : max_bytes_(max_pool_bytes),
          total_pages_(max_pool_bytes / KV_PAGE_BYTES),
          allocated_pages_(0) {
        
        cudaError_t err = cudaMallocManaged(&pool_base_ptr_, max_pool_bytes, cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            err = cudaMalloc(&pool_base_ptr_, max_pool_bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Failed to allocate KV pool: ") + cudaGetErrorString(err));
            }
        }

        free_pages_.reserve(total_pages_);
        pages_.resize(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            pages_[i].page_id = i;
            pages_[i].ref_count = 0;
            pages_[i].last_accessed_ns = 0;
            pages_[i].device_ptr = static_cast<char*>(pool_base_ptr_) + (i * KV_PAGE_BYTES);
            free_pages_.push_back(i);
        }
    }

    ~PagedKvPool() {
        if (pool_base_ptr_) {
            cudaFree(pool_base_ptr_);
        }
    }

    int32_t allocate_page() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_pages_.empty()) {
            return -1;
        }
        uint32_t page_id = free_pages_.back();
        free_pages_.pop_back();
        pages_[page_id].ref_count = 1;
        pages_[page_id].last_accessed_ns = current_time_ns();
        allocated_pages_++;
        return static_cast<int32_t>(page_id);
    }

    void free_page(uint32_t page_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (page_id >= total_pages_) return;
        pages_[page_id].ref_count--;
        if (pages_[page_id].ref_count <= 0) {
            pages_[page_id].ref_count = 0;
            free_pages_.push_back(page_id);
            allocated_pages_--;
        }
    }

    void* get_page_ptr(uint32_t page_id) const {
        if (page_id >= total_pages_) return nullptr;
        return pages_[page_id].device_ptr;
    }

    size_t get_used_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_pages_ * KV_PAGE_BYTES;
    }

    size_t get_free_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (total_pages_ - allocated_pages_) * KV_PAGE_BYTES;
    }

    size_t get_total_pages() const { return total_pages_; }
    size_t get_allocated_pages() const { return allocated_pages_; }

private:
    static uint64_t current_time_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    size_t max_bytes_;
    size_t total_pages_;
    size_t allocated_pages_;
    void* pool_base_ptr_{nullptr};
    std::vector<KvPage> pages_;
    std::vector<uint32_t> free_pages_;
    mutable std::mutex mutex_;
};

class GdnStateCache {
public:
    GdnStateCache(size_t max_snapshots = 512)
        : max_snapshots_(max_snapshots), snapshot_bytes_(GDN_TOTAL_STATE_BYTES) {
        size_t arena_size = max_snapshots_ * snapshot_bytes_;
        cudaError_t err = cudaMalloc(&arena_base_ptr_, arena_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GDN state cache arena");
        }

        for (size_t i = 0; i < max_snapshots_; ++i) {
            free_slots_.push_back(i);
        }
    }

    ~GdnStateCache() {
        if (arena_base_ptr_) {
            cudaFree(arena_base_ptr_);
        }
    }

    bool save_snapshot(uint64_t prefix_hash, const void* current_device_gdn_state, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(prefix_hash);
        size_t slot = 0;
        if (it != table_.end()) {
            slot = it->second.slot_id;
        } else {
            if (free_slots_.empty()) {
                uint64_t oldest_hash = 0;
                uint64_t oldest_time = UINT64_MAX;
                for (const auto& kv : table_) {
                    if (kv.second.last_access_ns < oldest_time) {
                        oldest_time = kv.second.last_access_ns;
                        oldest_hash = kv.first;
                    }
                }
                slot = table_[oldest_hash].slot_id;
                table_.erase(oldest_hash);
            } else {
                slot = free_slots_.back();
                free_slots_.pop_back();
            }
        }

        void* dest_ptr = static_cast<char*>(arena_base_ptr_) + (slot * snapshot_bytes_);
        cudaMemcpyAsync(dest_ptr, current_device_gdn_state, snapshot_bytes_, cudaMemcpyDeviceToDevice, stream);

        table_[prefix_hash] = { slot, current_time_ns() };
        return true;
    }

    bool restore_snapshot(uint64_t prefix_hash, void* target_device_gdn_state, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(prefix_hash);
        if (it == table_.end()) {
            return false;
        }

        it->second.last_access_ns = current_time_ns();
        void* src_ptr = static_cast<char*>(arena_base_ptr_) + (it->second.slot_id * snapshot_bytes_);
        cudaMemcpyAsync(target_device_gdn_state, src_ptr, snapshot_bytes_, cudaMemcpyDeviceToDevice, stream);
        return true;
    }

    bool has_snapshot(uint64_t prefix_hash) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.find(prefix_hash) != table_.end();
    }

    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.size();
    }

private:
    static uint64_t current_time_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    struct Entry {
        size_t slot_id;
        uint64_t last_access_ns;
    };

    size_t max_snapshots_;
    size_t snapshot_bytes_;
    void* arena_base_ptr_{nullptr};
    std::unordered_map<uint64_t, Entry> table_;
    std::vector<size_t> free_slots_;
    mutable std::mutex mutex_;
};

class MemoryGovernor {
public:
    MemoryGovernor(size_t total_system_bytes = GB10_TOTAL_MEMORY_BYTES)
        : total_system_bytes_(total_system_bytes),
          kv_pool_(std::make_unique<PagedKvPool>(KV_POOL_RESERVED_BYTES)),
          gdn_cache_(std::make_unique<GdnStateCache>(256)) {}

    PagedKvPool& kv_pool() { return *kv_pool_; }
    GdnStateCache& gdn_cache() { return *gdn_cache_; }

    double get_utilization_ratio() const {
        size_t used = WEIGHTS_RESERVED_BYTES + kv_pool_->get_used_bytes();
        return static_cast<double>(used) / static_cast<double>(total_system_bytes_);
    }

    bool can_admit_tokens(size_t prompt_tokens, size_t max_gen_tokens) const {
        size_t needed_pages = (prompt_tokens + max_gen_tokens + PAGE_SIZE_TOKENS - 1) / PAGE_SIZE_TOKENS;
        size_t needed_bytes = needed_pages * KV_PAGE_BYTES;
        return kv_pool_->get_free_bytes() >= needed_bytes;
    }

private:
    size_t total_system_bytes_;
    std::unique_ptr<PagedKvPool> kv_pool_;
    std::unique_ptr<GdnStateCache> gdn_cache_;
};

} // namespace pulse
