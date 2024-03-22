#include "model/gemma_lm.h"

namespace redisxlm {
namespace model {

std::string GemmaModel::_generate(const std::string& prompt_string, char* c_argv[],
                                  int argc) const {
    // auto const_argv = (const char**)c_argv;
    gcpp::LoaderArgs loader(argc, c_argv);

    // Rough heuristic for the number of threads to use
    size_t num_threads = static_cast<size_t>(
        std::clamp(static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    hwy::ThreadPool pool(num_threads);

    // Instantiate model and KV Cache
    gcpp::Gemma model(loader.tokenizer, loader.compressed_weights, loader.ModelType(), pool);
    auto kv_cache = CreateKVCache(loader.ModelType());
    size_t pos = 0;  // KV Cache position

    // Initialize random number generator
    std::mt19937 gen;
    std::random_device rd;
    gen.seed(rd());

    // Tokenize instruction
    std::vector<int> tokens = _tokenize(prompt_string, model.Tokenizer());
    size_t ntokens = tokens.size();

    std::string res;
    // This callback function gets invoked everytime a token is generated
    auto stream_token = [&res, &pos, &ntokens, tokenizer = model.Tokenizer()](int token, float) {
        ++pos;
        if (pos < ntokens) {
            // print feedback
        } else if (token != gcpp::EOS_ID) {
            std::string token_text;
            HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
            // todo: use stream pipe
            res += token_text;
        }
        return true;
    };

    GenerateGemma(model, _runtime_config, tokens, /*KV cache position = */ 0, kv_cache, pool,
                  stream_token, gen);

    return res;
}

std::vector<int> GemmaModel::_tokenize(
    const std::string& prompt_string,
    const sentencepiece::SentencePieceProcessor* tokenizer) const {
    std::string formatted
        = "<start_of_turn>user\n" + prompt_string + "<end_of_turn>\n<start_of_turn>model\n";
    std::vector<int> tokens;
    HWY_ASSERT(tokenizer->Encode(formatted, &tokens).ok());
    tokens.insert(tokens.begin(), 2);  // BOS token
    return tokens;
}

}  // namespace model
}  // namespace redisxlm