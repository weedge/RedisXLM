#include "module.h"

#include <cassert>

int RedisModule_OnLoad(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    assert(ctx != nullptr);

    return REDISMODULE_OK;
}
