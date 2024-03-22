#pragma once

#include "errors.h"

namespace redisxlm {
namespace model {

class BaseModel {
public:
    BaseModel() = default;
    virtual ~BaseModel() = default;

    std::string generate(const std::string& prompt_string, char* c_argv[], int argc) const {
        return _generate(prompt_string, c_argv, argc);
    }

private:
    virtual std::string _generate(const std::string& prompt_string, char* c_argv[], int argc) const
        = 0;
};

}  // namespace model
}  // namespace redisxlm
