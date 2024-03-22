#pragma once

#include <exception>
#include <string>

namespace redisxlm {
class Error : public std::exception {
public:
    Error() = default;  // eg: for make_unique make default obj unique ptr

    explicit Error(const std::string& msg) : _msg("ERR " + msg) {
    }

    Error(const Error&) = default;
    Error& operator=(const Error&) = default;

    Error(Error&&) = default;
    Error& operator=(Error&&) = default;

    virtual ~Error() = default;

    virtual const char* what() const noexcept {
        return _msg.data();
    }
    virtual bool is_empty() const noexcept {
        return _msg.size() == 0;
    }

private:
    std::string _msg;
};

class WrongArityError : public Error {
public:
    WrongArityError() : Error("WrongArity") {
    }

    WrongArityError(const WrongArityError&) = default;
    WrongArityError& operator=(const WrongArityError&) = default;

    WrongArityError(WrongArityError&&) = default;
    WrongArityError& operator=(WrongArityError&&) = default;

    virtual ~WrongArityError() = default;
};
}  // namespace redisxlm
