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

/// @file
/// @brief Wrapper for MPI functions that don't require a communicator.

#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/error_handling.hpp"

namespace kamping {

namespace internal {
/// @brief A global list of MPI data types registered to KaMPIng.
inline std::vector<MPI_Datatype> registered_mpi_types;
} // namespace internal

/// @brief Configuration for the behavior of the constructors and destructor of \ref kamping::Environment.
enum class InitMPIMode {
    InitFinalize,  ///< Call \c MPI_Init in the constructor of \ref Environment.
    NoInitFinalize ///< Do not call \c MPI_Init in the constructor of \ref Environment.
};

/// @brief Wrapper for MPI functions that don't require a communicator. If the template parameter `init_finalize_mode`
/// is set to \ref InitMPIMode::InitFinalize (default), \c MPI_Init is called in the constructor, and
/// \c MPI_Finalize is called in the destructor.
///
/// Note that \c MPI_Init and \c MPI_Finalize are global, meaning that if they are called on an Environment object they
/// must not be called again in any Environment object (or directly vie the \c MPI_* calls).
template <InitMPIMode init_finalize_mode = InitMPIMode::InitFinalize>
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

    /// @brief Calls MPI_Init without arguments and doesn't check whether MPI_Init has already been called.
    void init_unchecked() const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(NULL, NULL);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init with arguments and doesn't check whether MPI_Init has already been called.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    void init_unchecked(int& argc, char**& argv) const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(&argc, &argv);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init without arguments. Checks whether MPI_Init has already been called first.
    void init() const {
        if (initialized()) {
            return;
        }
        [[maybe_unused]] int err = MPI_Init(NULL, NULL);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init with arguments. Checks whether MPI_Init has already been called first.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    void init(int& argc, char**& argv) const {
        if (initialized()) {
            return;
        }
        [[maybe_unused]] int err = MPI_Init(&argc, &argv);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Finalize and frees all registered MPI data types.
    ///
    /// Even if you chose InitMPIMode::InitFinalize, you might want to call this function: As MPI_Finalize could
    /// potentially return an error, this function can be used if you want to be able to handle that error. Otherwise
    /// the destructor will call MPI_Finalize and not throw on any errors returned.
    void finalize() const {
        KASSERT(!finalized(), "Trying to call MPI_Finalize twice");
        free_registered_mpi_types();
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
    static double wtime() {
        return MPI_Wtime();
    }

    /// @brief Returns the resolution of Environment::wtime().
    ///
    /// @return The resolution in seconds.
    static double wtick() {
        return MPI_Wtick();
    }

    /// @brief The upper bound on message tags defined by the MPI implementation.
    /// @return The upper bound for tags.
    [[nodiscard]] static int tag_upper_bound() {
        int* tag_ub;
        int  flag;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag);
        KASSERT(flag, "Could not retrieve MPI_TAG_UB");
        return *tag_ub;
    }

    /// @brief Checks if the given tag is a valid message tag.
    /// @return Whether the tag is valid.
    [[nodiscard]] static bool is_valid_tag(int tag) {
        return tag >= 0 && tag <= tag_upper_bound();
    }

    /// @brief Register a new MPI data type to KaMPIng that will be freed when using Environment to finalize MPI.
    /// @param type The MPI data type to register.
    static void register_mpi_type(MPI_Datatype type) {
        internal::registered_mpi_types.push_back(type);
    }

    /// @brief Free all registered MPI data types.
    ///
    /// Only call this when you no longer want to use any MPI data types created by KaMPIng as other KaMPIng function
    /// will assume the created types still exist.
    static void free_registered_mpi_types() {
        for (auto type: internal::registered_mpi_types) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
            }
        }
        internal::registered_mpi_types.clear();
    }

    /// @brief Calls MPI_Finalize if finalize() has not been called before. Also frees all registered MPI data types.
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
                free_registered_mpi_types();
                // Just kassert the error code. We can't throw exceptions in the destructor.
                [[maybe_unused]] int err = MPI_Finalize();
                KASSERT(err == MPI_SUCCESS, "MPI_Finalize call failed.");
            }
        }
    }

}; // class Environment

/// @brief A global environment object to use when you don't want to create a new Environment object.
///
/// Note that \c inline \c const results in external linkage since C++17 (see
/// https://en.cppreference.com/w/cpp/language/inline).
inline Environment<InitMPIMode::NoInitFinalize> const mpi_env;

} // namespace kamping
