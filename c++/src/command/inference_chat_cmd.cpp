#include "command/inference_chat_cmd.h"

#include "model/gemma_lm.h"
#include "redisxlm.h"
#include "utils/strings.h"

namespace redisxlm {
namespace command {
/*
 *   REDISXLM.INFERENCE_CHAT --tokenizer ${TOKENIZER_PATH}/tokenizer.spm \
 *                          --compressed_weights ${MODEL_PATH}/2b-it-sfp.sbs \
 *                          --model 2b-it hello
 */
void InferenceChatCmd::_run(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) const {
    if (argc < 8) {
        throw WrongArityError();
    }

    auto* bc = RedisModule_BlockClient(ctx, _reply_func, _timeout_func, _free_func, 0);
    auto& g_instance = redisxlm::Redisxlm::thread_unsafety_instance();
    try {
        g_instance.worker_pool().enqueue(&InferenceChatCmd::_run_block_inference, this, bc, argv,
                                         argc);
    } catch (const Error& err) {
        RedisModule_AbortBlock(bc);
        RedisModule_ReplyWithError(ctx, err.what());
    }
}

void InferenceChatCmd::_run_block_inference(RedisModuleBlockedClient* bc, RedisModuleString** argv,
                                            int argc) const {
    assert(bc != nullptr);
    auto prompt_str = redisxlm::utils::to_string(argv[argc - 1]);
    auto c_argv = redisxlm::utils::to_new_char_argv(argv, argc);

    auto result = std::make_unique<AsyncResult>();
    try {
        // generate reply
        redisxlm::model::GemmaModel model;
        result->output = model.generate(prompt_str, c_argv, argc);
    } catch (const Error& err) {
        result->err = err;
    }

    for (int i = 0; i < argc; ++i) {
        delete[] c_argv[i];
    }
    delete[] c_argv;

    RedisModule_UnblockClient(bc, result.release());
}

int InferenceChatCmd::_reply_func(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);
    auto* res = static_cast<AsyncResult*>(RedisModule_GetBlockedClientPrivateData(ctx));
    assert(res != nullptr);

    if (!res->err.is_empty()) {
        return RedisModule_ReplyWithError(ctx, res->err.what());
    }

    RedisModule_Log(ctx, "debug", "reply res: %s", res->output.data());

    RedisModule_ReplyWithStringBuffer(ctx, res->output.data(), res->output.size());
    return REDISMODULE_OK;
}

int InferenceChatCmd::_timeout_func(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);
    RedisModule_Log(ctx, "debug", "timeout InferenceChatCmd");
    return RedisModule_ReplyWithNull(ctx);
}

void InferenceChatCmd::_free_func(RedisModuleCtx* ctx, void* privdata) {
    REDISMODULE_NOT_USED(ctx);
    RedisModule_Log(ctx, "debug", "free InferenceChatCmd");
    auto* result = static_cast<AsyncResult*>(privdata);
    delete result;
}

}  // namespace command
}  // namespace redisxlm