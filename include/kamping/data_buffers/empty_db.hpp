#pragma once

#include <vector>

template <typename T>
class EmptyDataBuffer {
public:
    EmptyDataBuffer() : _data(std::vector<T>()) {}

    auto begin() noexcept {
        return _data.begin();
    }

    auto end() noexcept {
        return _data.end();
    }

    T* data() {
        resize_if_needed();
        return _data.data();
    }

    void set_size(size_t to_size) {
        _size = to_size;
    }

    [[nodiscard]] size_t size() const {
        resize_if_needed();
        return _size;
    }

private:
    mutable std::vector<T> _data;
    size_t                 _size = 0;

    void resize_if_needed() const {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            _data.resize(_size);
        }
    }
};
