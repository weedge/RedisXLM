
#include "utils/worker_pool.h"

#include <cassert>

namespace redisxlm {
namespace utils {

WorkerPool::WorkerPool(const WorkerPoolOptions& opts) : _opts(opts), _quit(false) {
    for (auto idx = 0U; idx != opts.pool_size; ++idx) {
        _workers.emplace_back(std::thread([this]() { this->_run(); }));
    }
}

WorkerPool::~WorkerPool() {
    _stop();

    for (auto& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void WorkerPool::_run() {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait(lock, [this]() { return this->_quit || !this->_tasks.empty(); });

            if (_tasks.empty()) {
                assert(_quit);
                break;
            }

            task = std::move(_tasks.front());
            _tasks.pop();
        }

        task();
    }
}

void WorkerPool::_stop() {
    {
        std::lock_guard<std::mutex> lock(_mutex);

        _quit = true;
    }

    _cv.notify_all();
}

}  // namespace utils
}  // namespace redisxlm
