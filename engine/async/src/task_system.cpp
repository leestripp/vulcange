#include "vge/async/task_system.hpp"

#include <chrono>

#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

namespace vulcanGE::async {

namespace {
struct TlsWorker {
    const TaskSystem* pool = nullptr;
    unsigned index = 0;
};
thread_local TlsWorker tls_worker;
} // namespace

TaskSystem::TaskSystem(unsigned worker_count) : worker_count_(worker_count) {
    workers_.reserve(worker_count_);
    threads_.reserve(worker_count_);
    for (unsigned i = 0; i < worker_count_; ++i) {
        workers_.push_back(std::make_unique<Worker>(i));
    }
    for (unsigned i = 0; i < worker_count_; ++i) {
        // Capture the Worker pointer BY VALUE — capturing the local reference
        // `w` would dangle as soon as the loop iteration ends.
        Worker* w = workers_[i].get();
        threads_.emplace_back([this, w](std::stop_token stop) { worker_loop(*w, stop); });
        pin_to_core(i);
    }
}

TaskSystem::~TaskSystem() {
    wait_idle();
    for (auto& t : threads_) {
        t.request_stop();
    }
    wake_cv_.notify_all(); // after request_stop so sleepers observe it
    for (auto& t : threads_) {
        t.join(); // workers must fully exit before any member is destroyed
    }
}

std::optional<unsigned> TaskSystem::current_worker_index() const noexcept {
    if (tls_worker.pool != this) {
        return std::nullopt; // not a worker, or a worker of another pool
    }
    return tls_worker.index;
}

void TaskSystem::pin_to_core(unsigned index) noexcept {
#if defined(__linux__)
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0 || threads_.empty()) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<std::size_t>(index % hw), &set);
    pthread_setaffinity_np(threads_.back().native_handle(), sizeof(set), &set);
#else
    (void)index;
#endif
}

void TaskSystem::submit(Task task) {
    pending_.fetch_add(1, std::memory_order_relaxed);

    if (const auto idx = current_worker_index()) {
        // Owner thread: single-producer lock-free push into its own deque.
        workers_[*idx]->deque.push(std::move(task));
    } else {
        // External producer: mutex-guarded side channel, never the ring.
        const unsigned ext_idx =
            submit_cursor_.fetch_add(1, std::memory_order_relaxed) % worker_count_;
        workers_[ext_idx]->deque.push_external(std::move(task));
    }
    notify_one_worker();
}

void TaskSystem::notify_one_worker() {
    // Cheap signal; workers also poll before sleeping so a missed notify
    // only costs one idle quantum, never correctness.
    wake_cv_.notify_one();
}

TaskSystem::Task* TaskSystem::find_work(Worker& worker) {
    if (Task* node = worker.deque.pop()) {
        return node;
    }
    // Steal from a pseudo-random victim.
    const std::size_t start =
        (submit_cursor_.load(std::memory_order_relaxed) + worker.index * 7U) % worker_count_;
    for (std::size_t offset = 1; offset < worker_count_; ++offset) {
        Worker& victim = *workers_[(start + offset) % worker_count_];
        if (Task* node = victim.deque.steal()) {
            return node;
        }
    }
    return nullptr;
}

void TaskSystem::execute(Task* node) {
    std::unique_ptr<Task> guard(node);
    (*node)();
    guard.reset();
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard lock(wake_mutex_);
        wake_cv_.notify_all(); // wake wait_idle() sleepers
    }
}

void TaskSystem::worker_loop(Worker& worker, std::stop_token stop) {
    tls_worker = {this, worker.index};
    while (!stop.stop_requested()) {
        if (Task* node = find_work(worker)) {
            execute(node);
            continue;
        }
        if (pending_.load(std::memory_order_acquire) == 0) {
            std::unique_lock lock(wake_mutex_);
            wake_cv_.wait_for(lock, std::chrono::microseconds(200), [this, &stop] {
                return stop.stop_requested() ||
                       pending_.load(std::memory_order_acquire) > 0;
            });
        }
    }
    tls_worker = {};
}

void TaskSystem::wait_idle() {
    // Help drain: the calling thread participates (Plan §4.1).
    while (pending_.load(std::memory_order_acquire) > 0) {
        const unsigned idx = submit_cursor_.load(std::memory_order_relaxed) % worker_count_;
        bool did_work = false;
        for (std::size_t offset = 0; offset < worker_count_; ++offset) {
            if (Task* node = workers_[(idx + offset) % worker_count_]->deque.steal()) {
                execute(node);
                did_work = true;
                break;
            }
        }
        if (!did_work) {
            std::unique_lock lock(wake_mutex_);
            wake_cv_.wait(lock, [this] {
                return pending_.load(std::memory_order_acquire) == 0;
            });
        }
    }
}

void TaskSystem::parallel_for(std::size_t count, const std::function<void(std::size_t)>& fn) {
    if (count == 0) {
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        submit([&fn, i] { fn(i); });
    }
    wait_idle();
}

TaskSystem& default_pool() {
    static TaskSystem pool; // magic static: thread-safe init, leaked at exit
    return pool;
}

} // namespace vulcanGE::async
