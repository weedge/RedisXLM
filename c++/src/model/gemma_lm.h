#pragma once
#include "gemma/gemma.h"
#include "gemma/util/app.h"
#include "gemma/util/args.h"
#include "model/base_lm.h"

namespace redisxlm {
namespace model {

class GemmaModel : public BaseModel {
public:
    GemmaModel() {
        _runtime_config.max_tokens = 3072;
        _runtime_config.max_generated_tokens = 2048;
        _runtime_config.temperature = 1.0;
#ifdef DEBUG
        _runtime_config.verbosity = 2;
#endif
    }
    explicit GemmaModel(size_t max_tokens, size_t max_generated_tokens, float temperature) {
        _runtime_config.max_tokens = max_tokens ? max_tokens > 0 : 3072;
        _runtime_config.max_generated_tokens
            = max_generated_tokens ? max_generated_tokens > 0 : 2048;
        _runtime_config.temperature = temperature ? temperature > 0.0 : 1.0;
#ifdef DEBUG
        _runtime_config.verbosity = 2;
#endif
    }

private:
    std::string _generate(const std::string& prompt_string, char* c_argv[], int argc) const;
    std::vector<int> _tokenize(const std::string& prompt_string,
                               const sentencepiece::SentencePieceProcessor* tokenizer) const;

    gcpp::RuntimeConfig _runtime_config;
};

}  // namespace model
}  // namespace redisxlm