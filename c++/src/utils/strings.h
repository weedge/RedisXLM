#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "errors.h"
#include "redismodule.h"

namespace redisxlm {
namespace utils {

std::string_view to_string_view(RedisModuleString* str) {
    if (str == nullptr) {
        throw redisxlm::Error("null string");
    }

    std::size_t len = 0;
    auto* data = RedisModule_StringPtrLen(str, &len);
    return {data, len};
}

std::vector<std::string_view> to_string_views(RedisModuleString** argv, int argc) {
    assert(argv != nullptr && argc >= 0);

    std::vector<std::string_view> args;
    args.reserve(argc);
    for (auto idx = 0; idx < argc; ++idx) {
        args.push_back(to_string_view(argv[idx]));
    }

    return args;
}

std::string to_string(RedisModuleString* str) {
    if (str == nullptr) {
        throw Error("null string");
    }

    std::size_t len = 0;
    auto* data = RedisModule_StringPtrLen(str, &len);
    return {data, len};
}

char** to_new_char_argv(RedisModuleString** argv, int argc) {
    if (argv == nullptr || argc < 0) {
        throw Error("null string");
    }

    char** new_argv = new char*[argc];
    for (auto idx = 0; idx < argc; ++idx) {
        auto str = to_string(argv[idx]);
        new_argv[idx] = new char[str.length() + 1];
        std::strcpy(new_argv[idx], str.c_str());
    }

    return new_argv;
}

}  // namespace utils
}  // namespace redisxlm
