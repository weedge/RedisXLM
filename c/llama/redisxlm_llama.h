/*
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
/* 6.0 need open REDISMODULE_EXPERIMENTAL_API */
#define REDISMODULE_EXPERIMENTAL_API
#ifndef REDISXLM_LLAMA_H
#define REDISXLM_LLAMA_H
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/syscall.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

#include "../deps/ds/dict.h"
#include "../deps/ds/list.h"
#include "../deps/ds/skiplist.h"
#include "../deps/ds/util.h"
#include "../deps/threadpool/thpool.h"
#include "redismodule.h"

// define error
#define REDISXLM_LLAMA_ERRORMSG_SYNTAX "ERR syntax error"
#define REDISXLM_LLAMA_ERRORMSG_MGRT "ERR migrate error"
#define REDISXLM_LLAMA_ERRORMSG_DEL "ERR del error"
#define REDISXLM_LLAMA_ERRORMSG_CLI_DISCONN "ERR client disconnected error"

// define const
#define REDIS_LONGSTR_SIZE 42                   // Bytes needed for long -> str
#define REDIS_MGRT_CMD_PARAMS_SIZE 1024 * 1024  // send redis cmd params size
#define MAX_NUM_THREADS 128
#define REDISXLM_LLAMA_APIVER_1 1
/* redis version */
#define REDIS_VERSION 60000 /*6.0.0*/
/* REDISXLM_LLAMA version for linker */
#define REDISXLM_LLAMA_MAJOR 0
#define REDISXLM_LLAMA_MINOR 1
#define REDISXLM_LLAMA_PATCH 0
#define REDISXLM_LLAMA_SONAME 0.1.0

// define macro
#define UNUSED(V) ((void)V)
#define CREATE_CMD(name, tgt, attr, firstkey, lastkey, keystep)                         \
    do {                                                                                \
        if (RedisModule_CreateCommand(ctx, name, tgt, attr, firstkey, lastkey, keystep) \
            != REDISMODULE_OK) {                                                        \
            RedisModule_Log(ctx, "warning", "reg cmd error");                           \
            return REDISMODULE_ERR;                                                     \
        }                                                                               \
    } while (0);
#define CREATE_ROMCMD(name, tgt, firstkey, lastkey, keystep) \
    CREATE_CMD(name, tgt, "readonly", firstkey, lastkey, keystep);
#define CREATE_WRMCMD(name, tgt, firstkey, lastkey, keystep) \
    CREATE_CMD(name, tgt, "write deny-oom", firstkey, lastkey, keystep);

/* Using the following macro you can run code inside serverCron() with the
 * specified period, specified in milliseconds.
 * The actual resolution depends on server.hz. */
#define run_with_period(_ms_, _hz_) \
    if (((_ms_) <= 1000 / _hz_) || !(g_slots_meta_info.cronloops % ((_ms_) / (1000 / _hz_))))

#define ASYNC_LOCK(ctx)                             \
    do {                                            \
        if (g_slots_meta_info.async) {              \
            RedisModule_ThreadSafeContextLock(ctx); \
        }                                           \
    } while (0);
#define ASYNC_UNLOCK(ctx)                             \
    do {                                              \
        if (g_slots_meta_info.async) {                \
            RedisModule_ThreadSafeContextUnlock(ctx); \
        }                                             \
    } while (0);

// define struct type

//
void RedisxLMLlamaInit(RedisModuleCtx* ctx, int num_threads);
void RedisxLMLlamaFree(RedisModuleCtx* ctx);

#endif /* REDISXLM_LLAMA_H */