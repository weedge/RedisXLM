
#include "inference_chat_cmd.h"

#include "gemma/util/app.h"
#include "gemma/util/args.h"
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

    auto prompt_str = redisxlm::utils::to_string(argv[argc - 1]);
    auto c_argv = redisxlm::utils::to_new_char_argv(argv, argc);
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
    std::vector<int> tokens = _tokenize(prompt_str, model.Tokenizer());
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
            res += token_text;
        }
        return true;
    };

    GenerateGemma(
        model,
        {.max_tokens = 2048, .max_generated_tokens = 1024, .temperature = 1.0, .verbosity = 0},
        tokens, /*KV cache position = */ 0, kv_cache, pool, stream_token, gen);

    RedisModule_ReplyWithStringBuffer(ctx, res.data(), res.size());

    for (int i = 0; i < argc; ++i) {
        delete[] c_argv[i];
    }
    delete[] c_argv;
}

std::vector<int> InferenceChatCmd::_tokenize(
    const std::string& prompt_string,
    const sentencepiece::SentencePieceProcessor* tokenizer) const {
    std::string formatted
        = "<start_of_turn>user\n" + prompt_string + "<end_of_turn>\n<start_of_turn>model\n";
    std::vector<int> tokens;
    HWY_ASSERT(tokenizer->Encode(formatted, &tokens).ok());
    tokens.insert(tokens.begin(), 2);  // BOS token
    return tokens;
}

}  // namespace command
}  // namespace redisxlm