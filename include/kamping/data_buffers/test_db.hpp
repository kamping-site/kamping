#pragma once

#include "kamping/comm_helper/generic_helper.hpp"

template <typename Container>
class CustomDataBuffer {

public:
    explicit CustomDataBuffer(Container data) : _data(data){}

    using value_type = std::ranges::range_value_t<Container>;

    auto begin() noexcept {
        return _data.begin();
    }

    auto end() noexcept {
        return _data.end();
    }

    value_type* data() noexcept {
        return _data.data();
    }

    [[nodiscard]] size_t size() const noexcept {
        return _data.size();
    }

    void printContainer() {
        auto values = _data.get_data();
        std::cout << "Printing data buffer: " << std::endl;
        for (auto v : values) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    Container& underlying() {
        return _data;
    }


private:
    Container _data;
};

template <typename Buff>
concept HasPrint = requires(Buff buf) {
    {buf.printContainer()};
};


template <kamping::CommType type, typename SBuff, typename RBuff, typename Communicator>
requires HasPrint<RBuff>
void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    std::cout << "Print infer called " << std::endl;
    rbuf.printContainer();
    if constexpr (kamping::HasUnderlying<RBuff>) {
        infer<type>(sbuf, rbuf.underlying(), comm);
    }
}

