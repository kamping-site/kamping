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
#include "named_parameter_check_common.hpp"

int main(int /* argc */, char** /* argv */) {
#if defined(MISSING_REQUIRED_PARAMETER)
    testing::test_required_send_buf();
#elif defined(UNSUPPORTED_PARAMETER_NO_PARAMETERS)
    std::vector<int> v;
    testing::test_empty_arguments(kamping::send_buf(v));
#elif defined(UNSUPPORTED_PARAMETER_ONLY_OPTIONAL_PARAMETERS)
    std::vector<int> v;
    testing::test_optional_recv_buf(kamping::send_buf(v));
#elif defined(DUPLICATE_PARAMETERS)
    std::vector<int> v;
    testing::test_required_send_buf(kamping::send_buf(v), kamping::send_buf(v));
#else
    // If none of the above sections is active, this file will compile successfully.
#endif
}
