// This file is part of KaMPI.ng.
//
// Copyright 2021-2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_factories.hpp"

namespace kamping {


/// @brief Checks if all PEs provide the same value to this collective.
///
/// This collective function checks if all PEs have called it with the same value. The result is returned on all ranks.
/// @tparam Value Type of the value to check; must be comparable with `operator==`.
/// @param value The value of this rank. This value is compared with the ones provided by all other PEs.
/// @return On all ranks: `true` if all PEs have provided the same value, `false` otherwise.
template <typename Value>
bool Communicator::is_same_on_all_pes(Value const& value) {
    // TODO Assert that two values are comparable.
    static_assert(std::is_pod_v<Value>, "Value must be a POD type (more complex types are not implemented yet).");
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
        kamping::commutative);
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