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

#include "kamping/named_parameter_check.hpp"

namespace testing {
template <typename... Args>
void test_empty_arguments(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS());
}

template <typename... Args>
void test_required_send_buf(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS());
}

template <typename... Args>
void test_required_send_buf_optional_recv_buf(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(recv_buf));
}

template <typename... Args>
void test_optional_recv_buf(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(recv_buf));
}

template <typename... Args>
void test_required_send_recv_buf(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, recv_buf), KAMPING_OPTIONAL_PARAMETERS());
}

template <typename... Args>
void test_optional_send_recv_buf(Args...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(send_buf, recv_buf));
}

template <typename... Args>
void test_require_many_parameters(Args...) {
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, recv_buf, root, recv_counts, send_counts),
        KAMPING_OPTIONAL_PARAMETERS()
    );
}
} // namespace testing
