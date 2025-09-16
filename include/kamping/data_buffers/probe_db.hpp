#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/p2p/probe.hpp"

template <typename T>
class ProbeDataBuffer {
public:
    explicit ProbeDataBuffer() : _data(std::vector<T>()) {}

    using value_type = T;

    auto begin() noexcept {
        return _data.begin();
    }

    auto end() noexcept {
        return _data.end();
    }

    T* data() noexcept {
        return _data.data();
    }


    auto set_size(size_t size)noexcept {
        _size = size;
    }

    [[nodiscard]] size_t size() const {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            _data.resize(_size);
        }
        return _size;
    }

private:
    mutable std::vector<T> _data;
    size_t         _size = 0;
};
