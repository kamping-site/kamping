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
#include "kamping/kassert.hpp"

namespace kamping {

/// @brief Wrapper for MPI functions that don't require a communicator. If the template parameter `init_finalize` is set
/// to true (default), MPI_Init is called in the constructor, MPI_Finalize is called in the destructor.
template <bool init_finalize = true>
class Environment {
public:
    /// @brief Calls MPI_Init with arguments.
    ///
    /// @param argc The number of arguments
    /// @param argv The arguments
    Environment(int& argc, char**& argv) {
        if constexpr (init_finalize) {
            [[maybe_unused]] int err = MPI_Init(&argc, &argv);
            THROW_IF_MPI_ERROR(err, MPI_Init);
        }
    }

    /// @brief Calls MPI_Init without arguments.
    Environment() {
        if constexpr (init_finalize) {
            [[maybe_unused]] int err = MPI_Init(NULL, NULL);
            THROW_IF_MPI_ERROR(err, MPI_Init);
        }
    }

    /// @brief Checks whether MPI_Init has been called.
    ///
    /// @return Whether MPI_Init has been called.
    bool initialized() {
        int                  result;
        [[maybe_unused]] int err = MPI_Initialized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Initialized);
        return result == true;
    }

    /// @brief Checks whether MPI_Finalize has been called.
    ///
    /// @return Whether MPI_Finalize has been called.
    bool finalized() {
        int                  result;
        [[maybe_unused]] int err = MPI_Finalized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Finalized);
        return result == true;
    }

    /// @brief Calls MPI_Finalize
    ///
    /// As MPI_Finalize could potentially return an error, this function can be used if you want to be able to
    /// handle that error. Otherwise the destructor will call MPI_Finalize and not throw on any errors returned.
    void finalize() {
        KASSERT(!finalized(), "Trying to call MPI_Finalize twice");
        [[maybe_unused]] int err = MPI_Finalize();
        THROW_IF_MPI_ERROR(err, MPI_Finalize);
    }

    /// @brief Calls MPI_Finalize if finalize() has not been called before.
    ~Environment() {
        if constexpr (init_finalize) {
            bool is_already_finalized = false;
            try {
                is_already_finalized = finalized();
            } catch (MpiErrorException&) {
                // Just kassert. We can't throw exceptions in the destructor.
                KASSERT(false, "MPI_Finalized call failed.");
            }
            if (!is_already_finalized) {
                // Just kassert the error code. We can't throw exceptions in the destructor.
                [[maybe_unused]] int err = MPI_Finalize();
                KASSERT(err == MPI_SUCCESS, "MPI_Finalize call failed.");
            }
        }
    }
}; // class Environment

} // namespace kamping
