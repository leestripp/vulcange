#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace vulcanGE::async {

// ---------------------------------------------------------------------------
// Chase-Lev lock-free work-stealing deque (Plan §4.1).
// Tasks are heap-allocated move_only_function nodes referenced by pointer;
// a PMR pool allocator replaces the heap allocation in a later milestone.
//
// Capacity is a fixed power-of-two ring; when full, push() falls back to a
// mutex-guarded overflow queue so no work is ever dropped.
// ---------------------------------------------------------------------------
class WorkDeque {
public:
    using Task = std::move_only_function<void()>;

    WorkDeque() = default;
    WorkDeque(const WorkDeque&) = delete;
    WorkDeque& operator=(const WorkDeque&) = delete;

    // OWNER THREAD ONLY — the lock-free ring is single-producer.
    // External producers must use push_external().
    void push(Task task) {
        const std::size_t b = bottom_.load(std::memory_order_relaxed);
        const std::size_t t = top_.load(std::memory_order_acquire);
        if (b - t < kCapacity) {
            buffer_[b & kMask].store(new Task(std::move(task)), std::memory_order_relaxed);
            // Release store publishes the slot write (fence-free so TSan can
            // model the ordering — Plan §16.4 requires TSan-clean code).
            bottom_.store(b + 1, std::memory_order_release);
            return;
        }
        // Ring full -> overflow queue (rare; keeps the hot path lock-free).
        std::lock_guard lock(overflow_mutex_);
        overflow_.emplace_back(std::move(task));
    }

    // Any thread. Mutex-guarded overflow path — safe for concurrent
    // producers, off the hot path.
    void push_external(Task task) {
        std::lock_guard lock(overflow_mutex_);
        overflow_.emplace_back(std::move(task));
    }

    // Owner thread only. Returns nullptr when empty.
    [[nodiscard]] Task* pop() noexcept {
        std::size_t b = bottom_.load(std::memory_order_relaxed);
        if (b > 0) {
            b -= 1;
            // seq_cst store/load pair: StoreLoad ordering without a
            // standalone fence (TSan-compatible equivalent).
            bottom_.store(b, std::memory_order_seq_cst);
            std::size_t t = top_.load(std::memory_order_seq_cst); // CAS needs non-const lvalue
            if (t <= b) {
                Task* node = buffer_[b & kMask].load(std::memory_order_relaxed);
                if (t == b) { // last element: race with steal()
                    if (!top_.compare_exchange_strong(
                            t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                        node = nullptr;
                    }
                    bottom_.store(b + 1, std::memory_order_relaxed);
                }
                if (node != nullptr) {
                    return node;
                }
            } else {
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
        }
        return pop_overflow();
    }

    // Any thread. Returns nullptr when empty or on race loss.
    [[nodiscard]] Task* steal() noexcept {
        // seq_cst load pair: LoadLoad ordering without a standalone fence.
        std::size_t t = top_.load(std::memory_order_seq_cst);
        const std::size_t b = bottom_.load(std::memory_order_seq_cst);
        if (t >= b) {
            return nullptr;
        }
        Task* node = buffer_[t & kMask].load(std::memory_order_relaxed);
        if (!top_.compare_exchange_strong(
                t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return nullptr; // lost the race
        }
        return node;
    }

private:
    [[nodiscard]] Task* pop_overflow() noexcept {
        std::lock_guard lock(overflow_mutex_);
        if (overflow_.empty()) {
            return nullptr;
        }
        auto* node = new Task(std::move(overflow_.back()));
        overflow_.pop_back();
        return node;
    }

    static constexpr std::size_t kCapacity = 4096;
    static constexpr std::size_t kMask = kCapacity - 1;

    alignas(64) std::atomic<std::size_t> top_{0};
    alignas(64) std::atomic<std::size_t> bottom_{0};
    std::array<std::atomic<Task*>, kCapacity> buffer_{};

    std::mutex overflow_mutex_;
    std::vector<Task> overflow_;
};

} // namespace vulcanGE::async
