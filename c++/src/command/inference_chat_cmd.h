#pragma once

#include "command/base_cmd.h"
#include "gemma/gemma.h"

namespace redisxlm {
namespace command {

class InferenceChatCmd : public BaseCommand {
private:
    void _run(RedisModuleCtx* ctx, RedisModuleString** argv, int argc) const;

    std::vector<int> _tokenize(const std::string& prompt_string,
                               const sentencepiece::SentencePieceProcessor* tokenizer) const;
};
}  // namespace command
}  // namespace redisxlm