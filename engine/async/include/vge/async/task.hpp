#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

#include "vge/async/task_system.hpp"

namespace vulcanGE::async {

// ---------------------------------------------------------------------------
// task<T> — lazy coroutine task (Plan §4.1: coroutine-driven tasks).
// Awaiting a task uses symmetric transfer (no extra allocation or thread hop);
// use `co_await pool.schedule()` inside a task to hop onto the worker pool.
// ---------------------------------------------------------------------------
template <typename T>
class [[nodiscard]] task {
public:
    struct promise_type {
        std::optional<T> value;
        std::exception_ptr error;
        std::coroutine_handle<> continuation = std::noop_coroutine();

        [[nodiscard]] task get_return_object() noexcept {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        [[nodiscard]] std::suspend_always initial_suspend() const noexcept { return {}; }
        [[nodiscard]] auto final_suspend() const noexcept {
            struct FinalAwaiter {
                [[nodiscard]] bool await_ready() const noexcept { return false; }
                [[nodiscard]] std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) const noexcept {
                    return h.promise().continuation;
                }
                void await_resume() const noexcept {}
            };
            return FinalAwaiter{};
        }
        void unhandled_exception() noexcept { error = std::current_exception(); }
        template <typename U>
        void return_value(U&& v) noexcept(std::is_nothrow_constructible_v<T, U&&>) {
            value.emplace(std::forward<U>(v));
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit task(handle_type h) noexcept : handle_(h) {}
    task(task&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    task& operator=(task&& o) noexcept {
        if (this != &o) {
            destroy();
            handle_ = std::exchange(o.handle_, {});
        }
        return *this;
    }
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    ~task() { destroy(); }

    // Awaitable interface (symmetric transfer into the inner coroutine).
    [[nodiscard]] bool await_ready() const noexcept { return false; }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> outer) noexcept {
        handle_.promise().continuation = outer;
        return handle_;
    }
    T await_resume() {
        auto& p = handle_.promise();
        if (p.error) {
            std::rethrow_exception(p.error);
        }
        return std::move(*p.value);
    }

    // Start the task on the current thread without awaiting a result.
    void start() noexcept { handle_.resume(); }
    [[nodiscard]] bool done() const noexcept { return handle_.done(); }

private:
    void destroy() noexcept {
        if (handle_) {
            handle_.destroy();
            handle_ = {};
        }
    }

    handle_type handle_;
};

// void specialization.
template <>
class [[nodiscard]] task<void> {
public:
    struct promise_type {
        std::exception_ptr error;
        std::coroutine_handle<> continuation = std::noop_coroutine();

        [[nodiscard]] task get_return_object() noexcept {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        [[nodiscard]] std::suspend_always initial_suspend() const noexcept { return {}; }
        [[nodiscard]] auto final_suspend() const noexcept {
            struct FinalAwaiter {
                [[nodiscard]] bool await_ready() const noexcept { return false; }
                [[nodiscard]] std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) const noexcept {
                    return h.promise().continuation;
                }
                void await_resume() const noexcept {}
            };
            return FinalAwaiter{};
        }
        void unhandled_exception() noexcept { error = std::current_exception(); }
        void return_void() const noexcept {}
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit task(handle_type h) noexcept : handle_(h) {}
    task(task&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    task& operator=(task&& o) noexcept {
        if (this != &o) {
            destroy();
            handle_ = std::exchange(o.handle_, {});
        }
        return *this;
    }
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    ~task() { destroy(); }

    [[nodiscard]] bool await_ready() const noexcept { return false; }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> outer) noexcept {
        handle_.promise().continuation = outer;
        return handle_;
    }
    void await_resume() {
        if (handle_.promise().error) {
            std::rethrow_exception(handle_.promise().error);
        }
    }

    void start() noexcept { handle_.resume(); }
    [[nodiscard]] bool done() const noexcept { return handle_.done(); }

private:
    void destroy() noexcept {
        if (handle_) {
            handle_.destroy();
            handle_ = {};
        }
    }

    handle_type handle_;
};

// ---------------------------------------------------------------------------
// sync_wait — run a task (possibly hopping threads via co_await schedule())
// and block the calling thread until it completes. Test harness oriented;
// game code should prefer awaiting.
//
// Completion is signaled from final_suspend (not the coroutine body): once
// final_suspend's await_suspend runs, user code has finished executing and
// the frame is no longer being run by the worker, so the waiter may safely
// destroy it. Signaling in the body would race with the coroutine tail.
// ---------------------------------------------------------------------------
namespace detail {

class [[nodiscard]] sync_wait_signal {
public:
    struct promise_type {
        std::atomic<bool>* done = nullptr;
        std::exception_ptr error;

        [[nodiscard]] sync_wait_signal get_return_object() noexcept {
            return sync_wait_signal{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        [[nodiscard]] std::suspend_always initial_suspend() const noexcept { return {}; }
        [[nodiscard]] auto final_suspend() noexcept {
            struct FinalAwaiter {
                [[nodiscard]] bool await_ready() const noexcept { return false; }
                void await_suspend(std::coroutine_handle<promise_type> h) const noexcept {
                    // User code complete. The store MUST be the final frame
                    // access: once the waiter observes it, it may destroy the
                    // frame while this thread is still unwinding resume().
                    h.promise().done->store(true, std::memory_order_release);
                }
                void await_resume() const noexcept {}
            };
            return FinalAwaiter{};
        }
        void unhandled_exception() noexcept { error = std::current_exception(); }
        void return_void() const noexcept {}
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit sync_wait_signal(handle_type h) noexcept : handle_(h) {}
    sync_wait_signal(const sync_wait_signal&) = delete;
    sync_wait_signal& operator=(const sync_wait_signal&) = delete;
    ~sync_wait_signal() {
        if (handle_) {
            handle_.destroy();
        }
    }

    [[nodiscard]] promise_type& promise() noexcept { return handle_.promise(); }
    void start() noexcept { handle_.resume(); }

private:
    handle_type handle_;
};

} // namespace detail

template <typename T>
T sync_wait(TaskSystem& pool, task<T> t) {
    std::atomic<bool> done{false};

    std::optional<T> result;
    std::exception_ptr error;
    auto wrapper = [&]() -> detail::sync_wait_signal {
        co_await pool.schedule(); // hop onto the worker pool
        try {
            if constexpr (std::is_void_v<T>) {
                co_await std::move(t);
            } else {
                result = co_await std::move(t);
            }
        } catch (...) {
            error = std::current_exception();
        }
    };

    auto w = wrapper();
    w.promise().done = &done;
    w.start();

    // Spin on the release/acquire flag — brief by construction (the waiter is
    // a test/setup thread), and free of any missed-wakeup window.
    while (!done.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    if (w.promise().error) {
        std::rethrow_exception(w.promise().error);
    }
    if (error) {
        std::rethrow_exception(error);
    }
    if constexpr (!std::is_void_v<T>) {
        return std::move(*result);
    }
}

} // namespace vulcanGE::async
