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

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/error_handling.hpp"

namespace kamping {

enum InitMPIMode { InitFinalize, NoInitFinalize };

/// @brief Wrapper for MPI functions that don't require a communicator. If the template parameter `init_finalize` is set
/// to true (default), MPI_Init is called in the constructor, and MPI_Finalize is called in the destructor.
///
/// Note that MPI_Init and MPI_Finalize are global, meaning that if they are called on an Environment object they must
/// not be called again in any Environment object (or directly vie the MPI_* calls).
template <InitMPIMode init_finalize_mode = InitFinalize>
class Environment {
public:
    /// @brief Calls MPI_Init with arguments.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    Environment(int& argc, char**& argv) {
        if constexpr (init_finalize_mode == InitMPIMode::InitFinalize) {
            init(argc, argv);
        }
    }

    /// @brief Calls MPI_Init without arguments.
    Environment() {
        if constexpr (init_finalize_mode == InitMPIMode::InitFinalize) {
            init();
        }
    }

    /// @brief Calls MPI_Init without arguments.
    void init() const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(NULL, NULL);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init with arguments.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    void init(int& argc, char**& argv) const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(&argc, &argv);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Finalize
    ///
    /// Even if you chose InitMPIMode::InitFinalize, you might want to call this function: As MPI_Finalize could
    /// potentially return an error, this function can be used if you want to be able to handle that error. Otherwise
    /// the destructor will call MPI_Finalize and not throw on any errors returned.
    void finalize() const {
        KASSERT(!finalized(), "Trying to call MPI_Finalize twice");
        [[maybe_unused]] int err = MPI_Finalize();
        THROW_IF_MPI_ERROR(err, MPI_Finalize);
    }

    /// @brief Checks whether MPI_Init has been called.
    ///
    /// @return \c true if MPI_Init has been called, \c false otherwise.
    bool initialized() const {
        int                  result;
        [[maybe_unused]] int err = MPI_Initialized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Initialized);
        return result == true;
    }

    /// @brief Checks whether MPI_Finalize has been called.
    ///
    /// @return \c true if MPI_Finalize has been called, \c false otherwise.
    bool finalized() const {
        int                  result;
        [[maybe_unused]] int err = MPI_Finalized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Finalized);
        return result == true;
    }

    /// @brief Returns the elapsed time since an arbitrary time in the past.
    ///
    /// @return The elapsed time in seconds.
    double static wtime() {
        return MPI_Wtime();
    }

    /// @brief Returns the resolution of Environment::wtime().
    ///
    /// @return The resolution in seconds.
    double static wtick() {
        return MPI_Wtick();
    }

    /// @brief Calls MPI_Finalize if finalize() has not been called before.
    ~Environment() {
        if constexpr (init_finalize_mode == InitMPIMode::InitFinalize) {
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

/// @brief A global environment object to use when you don't want to create a new Environment object.
///
/// Because everything in Environment is const, it doesn't matter that every compilation unit will have its own copy of
/// this.
static const Environment<InitMPIMode::NoInitFinalize> mpi_env;

} // namespace kamping
