#pragma once
#include <memory>

namespace redisxlm {

// Base class for all native structures.
class Object : public std::enable_shared_from_this<Object> {
public:
    virtual ~Object() = default;
};

using ObjectSPtr = std::shared_ptr<Object>;

}  // namespace redisxlm