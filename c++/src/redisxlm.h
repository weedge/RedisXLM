#pragma once

#include "define.h"
#include "redismodule.h"
#include "utils/worker_pool.h"

namespace redisxlm {

class Redisxlm {
public:
    // redis load module to init this instance, runtime jus to read it
    static Redisxlm& thread_unsafety_instance() {
        static Redisxlm instance_;
        return instance_;
    }

    void init(RedisModuleCtx* ctx, RedisModuleString** argv, int argc);

    Redisxlm(const Redisxlm&) = delete;
    Redisxlm& operator=(const Redisxlm&) = delete;
    Redisxlm(Redisxlm&&) = delete;
    Redisxlm& operator=(Redisxlm&&) = delete;

    utils::WorkerPool& worker_pool() {
        return *_worker_pool;
    }

private:
    Redisxlm() = default;
    std::unique_ptr<utils::WorkerPool> _worker_pool;
};

}  // namespace redisxlm