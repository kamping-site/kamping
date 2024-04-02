// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/error_handling.hpp"
/// @addtogroup kamping_collectives
/// @{

/// @brief Perform a \c MPI_Barrier on this communicator.
///
/// Barrier takes no parameters. Any parameters passed will cause a compilation error.
///
/// The parameter pack prohibits the compiler from compiling this function when it's not used.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
void kamping::Communicator<DefaultContainerType, Plugins...>::barrier(Args... args) const {
    using namespace kamping::internal;
    static_assert(sizeof...(args) == 0, "You may not pass any arguments to barrier().");

    [[maybe_unused]] int err = MPI_Barrier(mpi_communicator());
    this->mpi_error_hook(err, "MPI_Barrier");
}
/// @}
