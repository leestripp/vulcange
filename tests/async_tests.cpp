#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include <vge/async/async.hpp>

using namespace vulcanGE::async;

TEST_CASE("TaskSystem: executes all submitted tasks exactly once") {
    TaskSystem pool(4);
    std::atomic<int> counter{0};
    constexpr int kTasks = 10000;
    for (int i = 0; i < kTasks; ++i) {
        pool.submit([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    pool.wait_idle();
    CHECK(counter.load() == kTasks);
}

TEST_CASE("TaskSystem: parallel_for sums across workers") {
    TaskSystem pool(TaskSystem::default_worker_count());
    constexpr std::size_t n = 100000;
    std::vector<std::int64_t> partial(n, 0);
    pool.parallel_for(n, [&partial](std::size_t i) {
        partial[i] = static_cast<std::int64_t>(i);
    });
    const std::int64_t total = std::accumulate(partial.begin(), partial.end(), std::int64_t{0});
    CHECK(total == static_cast<std::int64_t>(n * (n - 1) / 2));
}

TEST_CASE("TaskSystem: work-stealing drains a single-producer burst") {
    TaskSystem pool(TaskSystem::default_worker_count());
    std::atomic<int> counter{0};
    constexpr int kBurst = 5000;
    // All submitted from this (external) thread -> round-robin deques, workers steal.
    for (int i = 0; i < kBurst; ++i) {
        pool.submit([&counter] {
            // Small busy body to make stealing meaningful.
            int x = 0;
            for (int j = 0; j < 50; ++j) {
                x += j;
            }
            counter.fetch_add(x & 1, std::memory_order_relaxed); // deterministic: 1225 & 1 = 1
        });
    }
    pool.wait_idle();
    CHECK(counter.load() == kBurst); // each task contributes exactly 1
}

TEST_CASE("TaskSystem: tasks run on worker threads") {
    TaskSystem pool(4);
    std::atomic<bool> on_worker{false};
    const auto main_id = std::this_thread::get_id();
    pool.submit([&] {
        on_worker.store(pool.current_worker_index().has_value() &&
                            std::this_thread::get_id() != main_id,
                        std::memory_order_release);
    });
    pool.wait_idle();
    CHECK(on_worker.load());
}

TEST_CASE("task<T>: coroutine result propagates through pool hop") {
    TaskSystem pool(4);
    auto compute = [&pool]() -> task<int> {
        co_await pool.schedule(); // hop to a worker
        CHECK(pool.current_worker_index().has_value());
        co_return 21 * 2;
    };
    CHECK(sync_wait(pool, compute()) == 42);
}

TEST_CASE("task<T>: nested coroutine composition") {
    TaskSystem pool(4);
    auto inner = [&pool]() -> task<int> {
        co_await pool.schedule();
        co_return 10;
    };
    auto outer = [&pool](task<int> in) -> task<int> {
        co_await pool.schedule();
        const int v = co_await std::move(in);
        co_return v + 32;
    };
    CHECK(sync_wait(pool, outer(inner())) == 42);
}

TEST_CASE("task<T>: exceptions propagate to the awaiting caller") {
    TaskSystem pool(4);
    auto failing = [&pool]() -> task<int> {
        co_await pool.schedule();
        throw std::runtime_error("boom");
    };
    CHECK_THROWS_AS((void)sync_wait(pool, failing()), std::runtime_error);
}

TEST_CASE("dispatch_task: free-function API (Plan 11.3 Jolt bridge shape)") {
    std::atomic<int> counter{0};
    constexpr int kTasks = 2000;
    for (int i = 0; i < kTasks; ++i) {
        dispatch_task([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    default_pool().wait_idle();
    CHECK(counter.load() == kTasks);
}
