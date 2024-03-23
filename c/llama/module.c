/* redisxlm_llama module -- redis x llama inference
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

#include "cJSON.h"
#include "llama2.h"
#include "redisxlm_llama.h"

typedef struct {
    float temperature;
    float topp;
    int rng_seed;
    int steps;
} GenerateParams;

int pase_generate_params(const char* json_string, GenerateParams* param) {
    cJSON* json = cJSON_Parse(json_string);
    if (json == NULL) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        return REDISMODULE_ERR;
    }

    cJSON* temperature = cJSON_GetObjectItemCaseSensitive(json, "temperature");
    if (temperature) {
        param->temperature = temperature->valuedouble < 0.0 ? 0.0 : (float)temperature->valuedouble;
    }

    cJSON* rng_seed = cJSON_GetObjectItemCaseSensitive(json, "rng_seed");
    if (rng_seed) {
        param->rng_seed = rng_seed->valueint <= 0 ? (unsigned int)time(NULL) : rng_seed->valueint;
    }

    cJSON* topp = cJSON_GetObjectItemCaseSensitive(json, "topp");
    if (topp) {
        param->topp
            = (topp->valuedouble < 0.0 || 1.0 < topp->valuedouble) ? 0.9 : (float)topp->valuedouble;
    }

    cJSON* steps = cJSON_GetObjectItemCaseSensitive(json, "steps");
    if (steps) {
        param->steps = steps->valueint <= 0 ? (unsigned int)time(NULL) : steps->valueint;
    }

    cJSON_Delete(json);
    return REDISMODULE_OK;
}

/**
 * @brief
 * redisxllm.llama2.generate <checkpoint_path> <tokenizer_path> [prompt] [params]
 *
 * @param ctx
 * @param argv
 * @param argc
 * @return int
 */
int Llama2_Generate_RedisCommand(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    // 1. parse args
    if (argc < 3)
        return RedisModule_WrongArity(ctx);

    size_t m_len;
    const char* ck_path = RedisModule_StringPtrLen(argv[1], &m_len);

    size_t t_len;
    const char* tokenizer_path = RedisModule_StringPtrLen(argv[2], &t_len);

    size_t p_len;
    const char* prompt = "";
    if (argc >= 3) {
        prompt = RedisModule_StringPtrLen(argv[3], &p_len);
    }

    size_t s_len;
    const char* sample_params_str = "";
    if (argc >= 4) {
        sample_params_str = RedisModule_StringPtrLen(argv[4], &s_len);
    }
    GenerateParams params;
    if (pase_generate_params(sample_params_str, &params) == REDISMODULE_ERR) {
        return RedisModule_ReplyWithError(ctx, "ERR invalid generate params");
    }

    // 2. generate
    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, ck_path);
    if (params.steps == 0 || params.steps > transformer.config.seq_len)
        params.steps = transformer.config.seq_len;  // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, params.temperature, params.topp,
                  params.rng_seed);

    char* output = RedisModule_Alloc(sizeof(char) * transformer.config.seq_len);
    if (generate(&transformer, &tokenizer, &sampler, prompt, params.steps, output)
        == ERRNO_NUM_PROMPT_TOKENS) {
        return RedisModule_ReplyWithSimpleString(ctx, ERR_STR_NUM_PROMPT_TOKENS);
    }
    RedisModule_Log(ctx, "debug", "generate output len: %lu ", strlen(output));

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    return RedisModule_ReplyWithStringBuffer(ctx, output, strlen(output));
}

static int redisxlmLlamaInit(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    REDISMODULE_NOT_USED(ctx);
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);
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

    if (RedisModule_CreateCommand(ctx, "redisxllm.llama2.generate", Llama2_Generate_RedisCommand,
                                  "", 0, 0, 0)
        == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    return REDISMODULE_OK;
}

int RedisModule_OnUnload(RedisModuleCtx* ctx) {
    RedisxLMLlamaFree(ctx);
    return REDISMODULE_OK;
}
