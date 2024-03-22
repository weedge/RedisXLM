#pragma once

#include "command/base_cmd.h"

namespace redisxlm {
namespace command {

class InferenceChatCmd : public BaseCommand {
private:
    void _run(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) const;
    void _run_block_inference(RedisModuleBlockedClient* bc, RedisModuleString** argv,
                              int argc) const;

    struct AsyncResult {
        std::string output;
        Error err;
    };

    static int _reply_func(RedisModuleCtx* ctx, RedisModuleString** argv, int argc);
    static int _timeout_func(RedisModuleCtx* ctx, RedisModuleString** argv, int argc);
    static void _free_func(RedisModuleCtx* ctx, void* privdata);
};

}  // namespace command
}  // namespace redisxlm