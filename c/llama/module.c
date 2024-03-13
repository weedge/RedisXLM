/* slots migrate module -- batch dump rdb obj entries to migrate slots
 *
 * -----------------------------------------------------------------------------
 *
 * Copyright (c) 2024, weedge <weege007 at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "redisxlm_llama.h"

static int redisxlmLlamaInit(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    return REDISMODULE_OK;
}
/* This function must be present on each Redis module. It is used in
 * order to register the commands into the Redis server.
 *  __attribute__((visibility("default"))) for the same func name with redis
 * or other Dynamic Shared Lib *.so,  more detail man gcc or see
 * https://gcc.gnu.org/wiki/Visibility
 */
int __attribute__((visibility("default")))
RedisModule_OnLoad(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    if (RedisModule_Init(ctx, "redisxlm_llama_c", REDISXLM_LLAMA_APIVER_1, REDISMODULE_APIVER_1)
        == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    // Log the list of parameters passing loading the module.
    for (int j = 0; j < argc; j++) {
        const char* s = RedisModule_StringPtrLen(argv[j], NULL);
        printf("ModuleLoaded with argv[%d] = %s\n", j, s);
    }

    // init
    if (redisxlmLlamaInit(ctx, argv, argc) != REDISMODULE_OK) {
        printf("redisxlmLlamaInit fail! \n");
        return REDISMODULE_ERR;
    }

    // CREATE_WRMCMD("slotstest", SlotsDispatchRedisCommand, 0, 0, 0);

    return REDISMODULE_OK;
}

int RedisModule_OnUnload(RedisModuleCtx* ctx) {
    RedisxLMLlamaFree(ctx);
    return REDISMODULE_OK;
}
