#pragma once

#include "redismodule.h"

#ifdef __cplusplus

extern "C" {

#endif

int RedisModule_OnLoad(RedisModuleCtx* ctx, RedisModuleString** argv, int argc);

#ifdef __cplusplus
}

#endif
