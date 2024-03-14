#include "module.h"

#include <cassert>

static int redisxlmInit(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    return REDISMODULE_OK;
}
static void redisxlmFree(RedisModuleCtx* ctx) {
}

/* This function must be present on each Redis module. It is used in
 * order to register the commands into the Redis server.
 *  __attribute__((visibility("default"))) for the same func name with redis
 * or other Dynamic Shared Lib *.so,  more detail man gcc or see
 * https://gcc.gnu.org/wiki/Visibility
 */
int __attribute__((visibility("default")))
RedisModule_OnLoad(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    if (RedisModule_Init(ctx, "redisxlm_cpp", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    // Log the list of parameters passing loading the module.
    for (int j = 0; j < argc; j++) {
        const char* s = RedisModule_StringPtrLen(argv[j], NULL);
        printf("ModuleLoaded with argv[%d] = %s\n", j, s);
    }

    // init
    if (redisxlmInit(ctx, argv, argc) != REDISMODULE_OK) {
        printf("redisxlmLlamaInit fail! \n");
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}

int RedisModule_OnUnload(RedisModuleCtx* ctx) {
    redisxlmFree(ctx);
    return REDISMODULE_OK;
}
