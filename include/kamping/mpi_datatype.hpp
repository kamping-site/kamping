// This file is part of KaMPIng.
//
// Copyright 2021-2026 The KaMPIng Authors
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

// Forward declaration: fully defined below, after mpi_type_traits.
struct kamping_lookup;

/// @brief Maps a C++ type \p T to a type trait for constructing an MPI_Datatype.
///
/// Extends \ref kamping::types::type_dispatcher() with:
/// - All trivially copyable types not otherwise handled → `types::byte_serialized`.
///
/// \ref kamping_lookup is forwarded into \ref kamping::types::type_dispatcher() so that array and
/// enum branches resolve element types via the kamping layer (which includes the byte-serialization
/// fallback for trivially-copyable types).
///
/// @returns The corresponding type trait for the type \p T.
template <typename T>
auto type_dispatcher() {
    using T_no_const = std::remove_const_t<T>;
    if constexpr (types::has_auto_dispatched_type_v<T, kamping_lookup>) {
        return types::type_dispatcher<T, kamping_lookup>();
    } else if constexpr (std::is_trivially_copyable_v<T_no_const>) {
        return types::byte_serialized<T_no_const>{};
    } else {
        return internal::no_matching_type{};
    }
}

/// @brief Whether the type is handled by the auto-dispatcher \ref type_dispatcher,
/// i.e. whether \ref mpi_type_traits is defined without a user-provided specialization.
template <typename T>
static constexpr bool has_auto_dispatched_type_v =
    !std::is_same_v<decltype(type_dispatcher<T>()), internal::no_matching_type>;

/// @brief The type trait that maps a C++ type \p T to an MPI_Datatype for full KaMPIng.
///
/// The default behavior is controlled by \ref type_dispatcher. Specialize this trait in
/// `namespace kamping` to support additional types. Specializations of
/// \ref kamping::types::mpi_type_traits are intentionally ignored here.
template <typename T, typename Enable = void>
struct mpi_type_traits {};

/// @brief Partial specialization of \ref mpi_type_traits for types handled by \ref type_dispatcher.
template <typename T>
struct mpi_type_traits<T, std::enable_if_t<has_auto_dispatched_type_v<T>>> {
    /// @brief The base type of this trait obtained via \ref type_dispatcher.
    using base = decltype(type_dispatcher<T>());
    /// @brief The category of the type.
    static constexpr types::TypeCategory category = base::category;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = types::category_has_to_be_committed(category);
    /// @brief The MPI_Datatype corresponding to the type T.
    static MPI_Datatype data_type() {
        return decltype(type_dispatcher<T>())::data_type();
    }
};

/// @brief Check if the type has a static type definition, i.e. \ref kamping::mpi_type_traits is defined.
template <typename, typename Enable = void>
struct has_static_type : std::false_type {};

/// @brief Check if the type has a static type definition, i.e. \ref kamping::mpi_type_traits is defined.
template <typename T>
struct has_static_type<T, std::void_t<decltype(mpi_type_traits<T>::data_type())>> : std::true_type {};

/// @brief `true` if \ref kamping::mpi_type_traits provides a `data_type()` function.
template <typename T>
static constexpr bool has_static_type_v = has_static_type<T>::value;

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

/// @brief Lookup policy for KaMPIng that resolves MPI_Datatypes via \ref kamping::mpi_type_traits.
///
/// Unlike the module-level \ref kamping::types::type_dispatcher_lookup, this policy also covers
/// trivially-copyable types not matched by \ref kamping::types::type_dispatcher() (i.e., those
/// handled via `types::byte_serialized`).
struct kamping_lookup {
    /// @brief `true` if KaMPIng can resolve an MPI_Datatype for \p T.
    template <typename T>
    static constexpr bool has_type_v = has_static_type_v<T>;

    /// @brief Returns the MPI_Datatype for \p T.
    template <typename T>
    static MPI_Datatype get() {
        return mpi_type_traits<T>::data_type();
    }
};

// Backward-compatible aliases for types moved to kamping::types::
using types::builtin_type;
using types::byte_serialized;
using types::category_has_to_be_committed;
using types::is_builtin_type_v;
using types::kamping_tag;
using types::ScopedDatatype;
using types::type_dispatcher_lookup;
using types::TypeCategory;

/// @brief Constructs a contiguous MPI type of \p N elements of type \p T, using \ref kamping_lookup
/// to resolve element types (includes the byte-serialization fallback for trivially-copyable types).
/// @see kamping::types::contiguous_type
template <typename T, size_t N>
using contiguous_type = types::contiguous_type<T, N, kamping_lookup>;

/// @brief Constructs an MPI struct type for \p T, using \ref kamping_lookup to resolve field types
/// (includes the byte-serialization fallback for trivially-copyable types).
/// @see kamping::types::struct_type
template <typename T>
using struct_type = types::struct_type<T, kamping_lookup>;

/// @}

} // namespace kamping
