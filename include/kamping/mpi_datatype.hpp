// This file is part of KaMPIng.
//
// Copyright 2021-2025 The KaMPIng Authors
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
/// @brief Utility that maps C++ types to types that can be understood by MPI.

#pragma once

#include <type_traits>

#include <mpi.h>

#include "kamping/environment.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/noexcept.hpp"
#include "kamping/types/builtin_types.hpp"
#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/detail/type_helpers.hpp"
#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/scoped_datatype.hpp"
#include "kamping/types/struct_type.hpp"

namespace kamping {

/// @addtogroup kamping_mpi_utility
/// @{

// Re-export no_matching_type into kamping:: for backward compatibility
using internal::no_matching_type;

/// @brief Maps a C++ type \p T to a type trait for constructing an MPI_Datatype.
///
/// Extends \ref kamping::types::type_dispatcher() with:
/// - All trivially copyable types not otherwise handled → `byte_serialized`.
///
/// @returns The corresponding type trait for the type \p T.
template <typename T>
auto type_dispatcher() {
    using T_no_const = std::remove_const_t<T>;
    if constexpr (types::has_auto_dispatched_type_v<T>) {
        return types::type_dispatcher<T>();
    } else if constexpr (std::is_trivially_copyable_v<T_no_const>) {
        return byte_serialized<T_no_const>{};
    } else {
        return internal::no_matching_type{};
    }
}

/// @brief Partial specialization of \ref mpi_type_traits for trivially-copyable types not matched by
/// \ref type_dispatcher (i.e., types handled only via `byte_serialized`).
template <typename T>
struct mpi_type_traits<
    T,
    std::enable_if_t<!types::has_auto_dispatched_type_v<T> && std::is_trivially_copyable_v<std::remove_const_t<T>>>> {
    /// @brief The base type of this trait.
    using base = byte_serialized<std::remove_const_t<T>>;
    /// @brief The category of the type.
    static constexpr TypeCategory category = base::category;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = base::has_to_be_committed;
    /// @brief The MPI_Datatype corresponding to the type T.
    static MPI_Datatype data_type() {
        return base::data_type();
    }
};

/// @brief Whether the type is handled by the auto-dispatcher \ref type_dispatcher,
/// i.e. whether \ref mpi_type_traits is defined without a user-provided specialization.
template <typename T>
static constexpr bool has_auto_dispatched_type_v =
    !std::is_same_v<decltype(type_dispatcher<T>()), internal::no_matching_type>;

/// @brief Register a new \c MPI_Datatype for \p T with the MPI environment. It will be freed when the environment is
/// finalized.
///
/// The \c MPI_Datatype is created using \c mpi_type_traits<T>::data_type() and committed using \c MPI_Type_commit.
///
/// @tparam T The type to register.
template <typename T>
inline MPI_Datatype construct_and_commit_type() {
    MPI_Datatype type = mpi_type_traits<T>::data_type();
    MPI_Type_commit(&type);
    KAMPING_ASSERT(type != MPI_DATATYPE_NULL);
    mpi_env.register_mpi_type(type);
    return type;
}

/// @brief Translate type \p T to an MPI_Datatype using the type defined via \ref mpi_type_traits.
///
/// If the type has not been registered with MPI yet, it will be created and committed and automatically registered with
/// the MPI environment, such that it will be freed when the environment is finalized.
///
/// @tparam T The type to translate into an MPI_Datatype.
template <typename T>
[[nodiscard]] MPI_Datatype mpi_datatype() KAMPING_NOEXCEPT {
    static_assert(
        has_static_type_v<T>,
        "\n --> Type not supported directly by KaMPIng. Please provide a specialization for mpi_type_traits."
    );
    if constexpr (mpi_type_traits<T>::has_to_be_committed) {
        // using static initialization to ensure that the type is only committed once
        static MPI_Datatype type = construct_and_commit_type<T>();
        return type;
    } else {
        return mpi_type_traits<T>::data_type();
    }
}

/// @}

} // namespace kamping
