#pragma once

#include <vector>

template <typename T>
class EmptyDataBuffer {
public:
    EmptyDataBuffer() : _data(std::vector<T>()) {}

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

    std::vector<T>& get_data() {
        return _data;
    }

    auto set_size(size_t to_size) {
        _size = to_size;
    }

    [[nodiscard]] size_t size() {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            _data.resize(_size);
        }
        return _size;
    }

    [[nodiscard]] size_t size() const noexcept {
        return _data.size();
    }

private:
    std::vector<T> _data;
    size_t         _size = 0;
};
