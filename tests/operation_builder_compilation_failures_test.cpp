// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "kamping/operation_builder.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    OperationBuilder op_builder(ops::plus<>(), ops::commutative);
#if defined(COPY_CONSTRUCT_OP_BUILDER_BUFFER)
    // should not be possible to copy construct a buffer (for performance reasons)
    auto tmp = op_builder;
#elif defined(COPY_ASSIGN_OP_BUILDER_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    op_builder = op_builder;
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
