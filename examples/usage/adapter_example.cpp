// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>
#include <mdspan>
#include <vector>

#include "kamping/adapter/generic_adapter.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

template <typename Container>
bool correct_data_recv(Container const& sent, Container const& recv) {
    for (size_t i = 0; i < sent.size(); ++i) {
        if (sent[i] != recv[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    if (comm.size() <= 1) {
        std::cout << "Run with 2 ranks" << std::endl;
        return 0;
    }

    std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    if (comm.rank() == 0) {
        auto get_data = [](std::vector<int> const& vec) noexcept {
            return vec.data();
        };
        auto get_size = [](std::vector<int> const& vec) noexcept {
            return vec.size();
        };

        auto buff = adapter::generic_adapter_alt(v, get_data, get_size);
        comm.send(kamping::send_buf(buff), kamping::destination(1));
    } else if (comm.rank() == 1) {
        auto recv_data = comm.recv<int>(kamping::source(0));
        std::cout << "Using lambda without explicit T: " << correct_data_recv(v, recv_data) << std::endl;
    }

    comm.barrier();

    if (comm.rank() == 0) {
        auto get_data = [](std::vector<int> const& vec) noexcept {
            return vec.data();
        };
        auto get_size = [](std::vector<int> const& vec) noexcept {
            return vec.size();
        };

        auto buff = adapter::generic_adapter<int>(v, get_data, get_size);
        comm.send(kamping::send_buf(buff), kamping::destination(1));
    } else if (comm.rank() == 1) {
        auto recv_data = comm.recv<int>(kamping::source(0));
        std::cout << "Using lambda with explicit T: " << correct_data_recv(v, recv_data) << std::endl;
    }

    comm.barrier();

    if (comm.rank() == 0) {
        std::function<int const*(std::vector<int> const&)> get_data = [](std::vector<int> const& vec) noexcept {
            return vec.data();
        };
        std::function<size_t(std::vector<int> const&)> get_size = [](std::vector<int> const& vec) noexcept {
            return vec.size();
        };

        auto buff = adapter::generic_adapter_std_func(v, get_data, get_size);
        comm.send(kamping::send_buf(buff), kamping::destination(1));
    } else if (comm.rank() == 1) {
        auto recv_data = comm.recv<int>(kamping::source(0));
        std::cout << "Using std::function without explicit T: " << correct_data_recv(v, recv_data) << std::endl;
    }

    comm.barrier();

    auto mdspan_send   = std::mdspan(v.data(), 2, 6);
    auto get_data_span = [](decltype const(mdspan_send) & span) noexcept {
        return span.data_handle();
    };
    auto get_size_span = [](decltype const(mdspan_send) & span) noexcept {
        return span.size();
    };

    auto buff_span = adapter::generic_adapter<int>(mdspan_send, get_data_span, get_size_span);

    if (comm.rank() == 0) {
        comm.send(kamping::send_buf(buff_span), kamping::destination(1));
    } else if (comm.rank() == 1) {
        auto recv_data = comm.recv<int>(kamping::source(0));

        auto mdspan_recv = std::mdspan(recv_data.data(), 2, 6);

        bool is_same = true;
        for (std::size_t i = 0; i < mdspan_recv.extent(0); ++i) {
            for (std::size_t j = 0; j < mdspan_recv.extent(1); ++j) {
                is_same = mdspan_recv[i, j] == mdspan_send[i, j];
            }
        }

        std::cout << "Are the mdspans the same: " << is_same << std::endl;
    }
}
