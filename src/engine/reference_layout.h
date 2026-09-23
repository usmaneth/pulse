#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pulse {
// Copy logical elements from a tensor view. Dimension zero varies fastest.
inline std::vector<uint8_t> pack_reference(const uint8_t* data, size_t span,
        const int64_t ne[4], const size_t nb[4], size_t element_bytes) {
    size_t count = 1;
    for (int d = 0; d < 4; ++d) {
        if (ne[d] <= 0 || size_t(ne[d]) > SIZE_MAX / count)
            throw std::runtime_error("Invalid reference dimensions");
        count *= size_t(ne[d]);
    }
    if (!element_bytes || count > SIZE_MAX / element_bytes)
        throw std::runtime_error("Invalid reference element size");
    std::vector<uint8_t> packed(count * element_bytes);
    for (size_t i = 0; i < count; ++i) {
        size_t remainder = i, offset = 0;
        for (int d = 0; d < 4; ++d) {
            size_t coordinate = remainder % size_t(ne[d]);
            remainder /= size_t(ne[d]);
            if (coordinate && nb[d] > (SIZE_MAX - offset) / coordinate)
                throw std::runtime_error("Reference offset overflow");
            offset += coordinate * nb[d];
        }
        if (offset > span || element_bytes > span - offset)
            throw std::runtime_error("Reference view exceeds its byte span");
        std::memcpy(packed.data() + i * element_bytes, data + offset, element_bytes);
    }
    return packed;
}
}
