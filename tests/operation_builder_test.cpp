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

#include <gtest/gtest.h>

#include "kamping/operation_builder.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

TEST(OperationBuilderTest, move_constructor_assignment_operator_is_enabled) {
    // simply test that move ctor and assignment operator can be called.
    OperationBuilder op_builder1(ops::plus<>(), ops::commutative);
    OperationBuilder op_builder2(std::move(op_builder1));
    OperationBuilder op_builder3(ops::plus<>(), ops::commutative);
    op_builder3 = std::move(op_builder2);
}
