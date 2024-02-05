// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
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

#include <cassert>
#include <complex>
#include <cstdint>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>
#include <pfr.hpp>

#include "./datatype_detail/builtin.hpp"
#include "./datatype_detail/traits.hpp"
#include "./datatype_detail/tuple_like.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/environment.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/noexcept.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

template <typename T>
inline MPI_Datatype construct_and_commit_type() {
    MPI_Datatype type = mpi_type_traits<T>::data_type();
    MPI_Type_commit(&type);
    KASSERT(type != MPI_DATATYPE_NULL);
    mpi_env.register_mpi_type(type);
    return type;
}

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new custom continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// To check if type \c T maps to a builtin \c MPI_Datatype at compile-time, use \c mpi_type_traits.
/// @tparam T The type to translate into an MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///
template <typename T>
[[nodiscard]] MPI_Datatype mpi_datatype() KAMPING_NOEXCEPT {
    if constexpr (mpi_type_traits<T>::category == TypeCategory::kamping_provided || mpi_type_traits<T>::category == TypeCategory::user_provided) {
        static MPI_Datatype type = construct_and_commit_type<T>();
        return type;
    } else {
        return mpi_type_traits<T>::data_type();
    }

    // Remove const qualifiers.
    // Previously, we also removed volatile qualifiers here. MPI does not support volatile pointers and
    // removing volatile from a pointer is undefined behavior.
    using T_no_const = std::remove_const_t<T>;

    // Check if we got a pointer type -> error
    static_assert(
        !std::is_pointer_v<T_no_const>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?"
    );

    // Check if we got a function type -> error
    static_assert(!std::is_function_v<T_no_const>, "MPI does not support function types.");

    // Check if we got a union type -> error
    static_assert(!std::is_union_v<T_no_const>, "MPI does not support union types.");

    // Check if we got void -> error
    static_assert(!std::is_void_v<T_no_const>, "There is no MPI datatype corresponding to void.");

    // // Check if we got an array type -> create a continuous type.
    // if constexpr (std::is_array_v<T_no_cv>) {
    //     // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
    //     return mpi_custom_continuous_type<sizeof(T_no_cv)>();
    // }

    // // // Check if we got an enum type -> use underlying type
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    // if constexpr (std::is_enum_v<T_no_cv>) {
    //     return mpi_datatype<std::underlying_type_t<T_no_cv>>();
    // } else {
    //     static_assert(expression, );
    //     mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    // }
    // static_assert(std::is_enum_v<T_no_const>, "Type not supported");

    KASSERT(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}
/// @}

} // namespace kamping
