#pragma once

#include "errors.h"
#include "redismodule.h"

namespace redisxlm {
namespace command {
class BaseCommand {
public:
    virtual ~BaseCommand() = default;
    int run(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) const {
        try {
            _run(ctx, argv, argc);
        } catch (const WrongArityError&) {
            return RedisModule_WrongArity(ctx);
        } catch (const Error& err) {
            RedisModule_Log(ctx, "warning", "%s", err.what());
            return REDISMODULE_ERR;
        }
        return REDISMODULE_OK;
    }

private:
    virtual void _run(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) const = 0;
};
}  // namespace command
}  // namespace redisxlm
