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

#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_factories.hpp"

namespace kamping {

/// @brief Checks if all ranks provide the same value to this collective.
///
/// This collective function checks if all ranks have called it with the same value. The result is returned on all
/// ranks.
/// @tparam Value Type of the value to check. Must be comparable with `operator==`.
/// @param value The value of this rank. This value is compared with the ones provided by all other ranks.
/// @return `true` if all ranks have provided the same value, `false` otherwise.
template <typename Value>
bool Communicator::is_same_on_all_ranks(Value const& value) const {
    // TODO Assert that two values are comparable.
    // std::pair<> is not trivially_copyable and we don't want to forbid comparing std::pair<>s for equality.
    /// @todo How to handle more complex data types, e.g. std::pair<>, user defined classes, std::vector (here and
    /// elsewhere)?
    // static_assert(
    //     std::is_trivially_copyable_v<Value>,
    //     "Value must be a trivial type (more complex types are not implemented yet).");
    static_assert(!std::is_pointer_v<Value>, "Comparing pointers from different machines does not make sense.");

    /// @todo Expose this functionality to the user, he might find it useful, too.
    /// @todo Implement this for complex types.

    struct ValueEqual {
        Value value; // The value to compare, init on each rank with the local value.
        bool  equal; // Have we seen only equal values in the reduction so far?
    };
    ValueEqual value_equal = {value, true};
    const auto datatype    = mpi_datatype<ValueEqual>();

    // Build the operation for the reduction.
    auto operation_param = kamping::op(
        [](auto a, auto b) {
            if (a.equal && b.equal && a.value == b.value) {
                return ValueEqual{a.value, true};
            } else {
                return ValueEqual{a.value, false};
            }
        },
        kamping::commutative
    );
    auto operation = operation_param.template build_operation<ValueEqual>();

    // Perform the reduction and return.
    MPI_Allreduce(
        MPI_IN_PLACE,            // sendbuf
        &value_equal,            // recvbuf
        1,                       // count
        datatype,                // datatype,
        operation.op(),          // operation,
        this->mpi_communicator() // communicator
    );

    return value_equal.equal;
}

} // namespace kamping
