#pragma once

#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/p2p/probe.hpp"

template <typename T>
class ProbeDataBuffer {
public:
    explicit ProbeDataBuffer(int source) : _data(std::vector<T>()), _source(source) {}

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

    template <typename Communicator>
    auto set_size(Communicator& comm) {
        auto status = comm.probe(kamping::status_out()).extract_status();
        _size = kamping::asserting_cast<size_t>(status.template count_signed<int>());
    }

    [[nodiscard]] size_t size() const /* noexcept */ {
        size_t curr_size = _data.size();
        if (curr_size != _size) {
            resize();
        }
        return _size;
    }

    void resize() {
        _data.resize(_size);
    }

    // [[nodiscard]] size_t size() const noexcept {
    //     return _data.size();
    // }

    [[nodiscard]] int source() const noexcept {return _source;}

private:
    mutable std::vector<T> _data;
    int _source;
    size_t         _size = 0;
};
