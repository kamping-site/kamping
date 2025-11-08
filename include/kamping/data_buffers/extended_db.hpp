#pragma once

#include <span>

class ExtDataBuffer {
public:
    ExtDataBuffer(std::vector<int>& data)
        : _data(data),
          _displs(std::vector<int>()),
          _sizes(std::vector<int>()),
          _size(_data.size()) {}

    auto begin() noexcept {
        return _data.begin();
    }

    auto end() noexcept {
        return _data.end();
    }

    auto data() {
        resize_if_needed();
        return _data.data();
    }

    [[nodiscard]] size_t size() const {
        resize_if_needed();
        return _size;
    }

    [[nodiscard]] std::vector<int> const& size_v() const noexcept {
        return _sizes;
    }

    [[nodiscard]] std::vector<int> const& displs() const noexcept {
        return _displs;
    }

    void set_size_v(std::vector<int>&& sizes) {
        _sizes = std::move(sizes);
    }

    void set_displs(std::vector<int>&& displs) {
        _displs = std::move(displs);
    }

    void set_size(size_t size) {
        _size = size;
    }

private:
    mutable std::vector<int> _data;
    std::vector<int>         _displs;
    std::vector<int>         _sizes;
    size_t                   _size = 0;

    void resize_if_needed() const {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            _data.resize(_size);
        }
    }
};
