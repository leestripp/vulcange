#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "vge/async/work_deque.hpp"

namespace vulcanGE::async {

// ---------------------------------------------------------------------------
// TaskSystem — lock-free work-stealing fiber pool (Plan §4.1).
// N-1 std::jthread workers (pinned to cores on Linux) plus the calling
// thread participates in drain points. Tasks are std::move_only_function.
// ---------------------------------------------------------------------------
class TaskSystem {
public:
    using Task = WorkDeque::Task;

    explicit TaskSystem(unsigned worker_count = default_worker_count());
    ~TaskSystem(); // drains and joins all workers

    TaskSystem(const TaskSystem&) = delete;
    TaskSystem& operator=(const TaskSystem&) = delete;

    // Submit from any thread. Thread-safe.
    void submit(Task task);

    // Block until every submitted task has completed.
    void wait_idle();

    // Execute fn for indices [0, count) scattered across the pool, then block
    // until all complete.
    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& fn);

    [[nodiscard]] unsigned worker_count() const noexcept { return worker_count_; }

    // Awaitable: co_await pool.schedule() resumes the coroutine on a worker.
    [[nodiscard]] auto schedule() noexcept {
        struct ScheduleAwaitable {
            TaskSystem& pool;
            [[nodiscard]] bool await_ready() const noexcept { return false; }
            void await_suspend(std::coroutine_handle<> h) const {
                pool.submit([h]() mutable { h.resume(); });
            }
            void await_resume() const noexcept {}
        };
        return ScheduleAwaitable{*this};
    }

    // If called on a worker thread OF THIS POOL, returns its index;
    // otherwise nullopt. (Cross-pool workers are treated as external.)
    [[nodiscard]] std::optional<unsigned> current_worker_index() const noexcept;

    [[nodiscard]] static unsigned default_worker_count() noexcept {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 2 ? hw - 1 : 1;
    }

private:
    struct Worker {
        explicit Worker(unsigned idx) : index(idx) {}
        WorkDeque deque;
        unsigned index;
    };

    void worker_loop(Worker& worker, std::stop_token stop);
    Task* find_work(Worker& worker);
    void execute(Task* node);
    void notify_one_worker();
    void pin_to_core(unsigned index) noexcept;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<std::jthread> threads_;

    std::atomic<std::int64_t> pending_{0};
    std::atomic<unsigned> submit_cursor_{0};

    // Sleep/wake: one cv guards both worker sleep and wait_idle.
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;

    unsigned worker_count_;
};

// Global default pool (lazy). Used by the Jolt bridge (Plan §11.3) and by
// dispatch_task below.
[[nodiscard]] TaskSystem& default_pool();

// Free-function dispatch matching Plan §11.3: vulcanGE::async::dispatch_task.
template <typename F>
void dispatch_task(F&& fn) {
    default_pool().submit(TaskSystem::Task(std::forward<F>(fn)));
}

} // namespace vulcanGE::async
