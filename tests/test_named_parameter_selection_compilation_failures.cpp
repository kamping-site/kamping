// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "kamping/named_parameter_selection.hpp"

#include "helpers_for_testing.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    testing::Argument<ParameterType::send_buf> arg0{0};
    testing::Argument<ParameterType::recv_buf> arg1{1};
    // if the requested ParameterType is not given, parameter selection should fail to compile.
#if defined(REQUESTED_PARAMETER_NOT_GIVEN)
    const auto& selected_arg = select_parameter_type<ParameterType::send_counts>(arg0, arg1);
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
