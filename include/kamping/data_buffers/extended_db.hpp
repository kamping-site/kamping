#pragma once

#include <span>

class ExtDataBuffer {
public:

    ExtDataBuffer(std::vector<int>& data)
        : _data(data),
          _displacements(std::vector<int>()),
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

    [[nodiscard]] std::vector<int> const& displacements() const noexcept {
        return _displacements;
    }

    void set_size_v(std::vector<int>&& sizes) {
        _sizes = std::move(sizes);
    }

    void set_displacements(std::vector<int>&& displacements) {
        _displacements = std::move(displacements);
    }

    void set_size(size_t size) {
        _size = size;
    }

private:
    mutable std::vector<int> _data;
    std::vector<int>         _displacements;
    std::vector<int>         _sizes;
    size_t                   _size = 0;

    void resize_if_needed() const {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            _data.resize(_size);
        }
    }
};
