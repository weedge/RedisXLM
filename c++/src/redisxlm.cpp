#include "redisxlm.h"

#include <memory>
#include <string>
#include <thread>

namespace redisxlm {

void Redisxlm::init(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    RedisModule_Log(ctx, "debug", "start Redisxlm::init...");

    // worker_pool num_threads
    long long num_threads = 0;
    if (argc >= 1 && RedisModule_StringToLongLong(argv[0], &num_threads) == REDISMODULE_ERR) {
        throw Error("the 1st params [worker_pool num_threads] must int");
    }
    if (num_threads < 0) {
        throw Error("the 1st params [worker_pool num_threads] must >0");
    }
    if (num_threads > WP_MAX_THREAD_NUM) {
        throw Error("the 1st params [worker_pool num_threads] must <="
                    + std::to_string(WP_MAX_THREAD_NUM));
    }
    if (num_threads == 0) {
        num_threads = (long long)std::thread::hardware_concurrency();
    }
    RedisModule_Log(ctx, "debug", "[worker_pool num_threads]=%d", num_threads);

    // worker_pool task queue_size
    long long queue_size = 0;
    if (argc >= 2 && RedisModule_StringToLongLong(argv[1], &queue_size) == REDISMODULE_ERR) {
        throw Error("the 1st params [worker_pool queue_size] must int");
    }
    if (queue_size < 0) {
        throw Error("the 1st params [worker_pool queue_size] must >0");
    }
    if (queue_size > WP_MAX_QUEUE_SIZE) {
        throw Error("the 1st params [worker_pool queue_size] must <="
                    + std::to_string(WP_MAX_QUEUE_SIZE));
    }
    if (queue_size == 0) {
        queue_size = WP_MAX_QUEUE_SIZE;
    }
    RedisModule_Log(ctx, "debug", "[worker_pool queue_size]=%d", queue_size);

    const utils::WorkerPoolOptions options = {
        .pool_size = (std::size_t)num_threads,
        .queue_size = (std::size_t)queue_size,
    };
    _worker_pool = std::make_unique<utils::WorkerPool>(options);
}

}  // namespace redisxlm