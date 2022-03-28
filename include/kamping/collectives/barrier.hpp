// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
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

#include <mpi.h>

#include "kamping/error_handling.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"

namespace kamping::internal {
/// @brief CRTP mixin class for \c MPI_Barrier.
///
/// This class is only to be used as a super class of kamping::Communicator
template <typename Communicator>
class Barrier : public CRTPHelper<Communicator, Barrier> {
public:
    /// @brief Wrapper for \c MPI_Barrier.
    ///
    /// Perform a \c MPI_Barrier on this communicator.
    /// buffers are required:
    void barrier() {
        [[maybe_unused]] int err = MPI_Barrier(this->underlying().mpi_communicator());
        THROW_IF_MPI_ERROR(err, MPI_Barrier);
    }

protected:
    Barrier() {}
};
} // namespace kamping::internal
