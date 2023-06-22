// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include "kamping/named_parameters.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using DataType = size_t;
    DataType single_data{};
    using ContainerType = std::vector<DataType>;
    ContainerType container;

#if defined(RECV_COUNT_OUT_PASSED)
    // should not be possible create a recv_count_out buffer with a type different than int
    auto tmp = recv_count_out(single_data);
#elif defined(RECV_COUNT_OUT_NEW_CONTAINER)
    // should not be possible create a recv_count_out buffer with a type different than int with alloc_new
    auto tmp = recv_count_out(alloc_new<DataType>{});
#elif defined(RECV_COUNTS_PASSED)
    // should not be possible create a recv_counts buffer with a type different than int
    auto tmp = recv_counts(container);
#elif defined(RECV_COUNTS_OUT_PASSED)
    // should not be possible create a recv_counts_out buffer with a type different than int
    auto tmp = recv_counts(container);
#elif defined(RECV_COUNTS_OUT_NEW_CONTAINER)
    // should not be possible create a recv_counts_out buffer with a type different than int with alloc_new
    auto tmp = recv_counts_out(alloc_new<ContainerType>{});
#elif defined(SEND_COUNTS_PASSED)
    // should not be possible create a send_counts buffer with a type different than int
    auto tmp = send_counts(container);
#elif defined(RECV_DISPLS_PASSED)
    // should not be possible create a recv_displs buffer with a type different than int
    auto tmp = recv_displs(container);
#elif defined(RECV_DISPLS_OUT_PASSED)
    // should not be possible create a recv_displs_out buffer with a type different than int
    auto tmp = recv_displs_out(container);
#elif defined(RECV_DISPLS_OUT_NEW_CONTAINER)
    // should not be possible create a recv_displs_out buffer with a type different than int with alloc_new
    auto tmp = recv_displs_out(alloc_new<ContainerType>{});
#elif defined(SEND_DISPLS_PASSED)
    // should not be possible create a send_displs buffer with a type different than int
    auto tmp = send_displs(container);
#elif defined(SEND_DISPLS_OUT_PASSED)
    // should not be possible create a send_displs_out buffer with a type different than int
    auto tmp = send_displs_out(container);
#elif defined(SEND_DISPLS_OUT_NEW_CONTAINER)
    // should not be possible create a send_displs_out buffer with a type different than int with alloc_new
    auto tmp = send_displs_out(alloc_new<ContainerType>{});
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
