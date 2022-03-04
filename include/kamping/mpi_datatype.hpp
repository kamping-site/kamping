// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Utility that maps C++ types to types that can be understood by MPI.

#include <cassert>
#include <complex>
#include <cstdint>
#include <type_traits>

#include <mpi.h>

#include "kamping/kassert.hpp"

namespace kamping {

/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Creates a custom continuous MPI datatype.
///
/// @tparam NumBytes The number of bytes for the new type.
/// @return The newly created MPI_Datatype.
/// @see mpi_datatype()
///
template <size_t NumBytes>
[[nodiscard]] MPI_Datatype mpi_custom_continuous_type() noexcept {
    static_assert(NumBytes > 0, "You cannot create a continuous type with 0 bytes.");
    // Create a new MPI datatype only the first type per NumBytes this function is called.
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(NumBytes, MPI_CHAR, &type);
        MPI_Type_commit(&type);
        assert(type != MPI_DATATYPE_NULL);
    }
    // From the second call onwards, re-use the existing type.
    return type;
}

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new custom continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// @tparam T The type to translate into a MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_datatype() noexcept {
    // Remove const and volatile qualifiers.
    using T_no_cv = std::remove_cv_t<T>;

    // Check if we got a pointer type -> error
    static_assert(
        !std::is_pointer_v<T_no_cv>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?");

    // Check if we got a function type -> error
    static_assert(!std::is_function_v<T_no_cv>, "MPI does not support function types.");

    // Check if we got a union type -> error
    static_assert(!std::is_union_v<T_no_cv>, "MPI does not support union types.");

    // Check if we got void -> error
    static_assert(!std::is_void_v<T_no_cv>, "There is no MPI datatype corresponding to void.");

    // Check if we got a array type -> create a continuous type.
    if constexpr (std::is_array_v<T_no_cv>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        return mpi_custom_continuous_type<sizeof(T_no_cv)>();
    }

    // Check if we got a enum type -> use underlying type
    if constexpr (std::is_enum_v<T_no_cv>) {
        return mpi_datatype<std::underlying_type_t<T_no_cv>>();
    }

    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if constexpr (std::is_same_v<T_no_cv, char>) {
        mpi_type = MPI_CHAR;
    } else if constexpr (std::is_same_v<T_no_cv, signed char>) {
        mpi_type = MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same_v<T_no_cv, unsigned char>) {
        mpi_type = MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T_no_cv, wchar_t>) {
        mpi_type = MPI_WCHAR;
    } else if constexpr (std::is_same_v<T_no_cv, signed short>) {
        mpi_type = MPI_SHORT;
    } else if constexpr (std::is_same_v<T_no_cv, unsigned short>) {
        mpi_type = MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<T_no_cv, signed int>) {
        mpi_type = MPI_INT;
    } else if constexpr (std::is_same_v<T_no_cv, unsigned int>) {
        mpi_type = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T_no_cv, signed long int>) {
        mpi_type = MPI_LONG;
    } else if constexpr (std::is_same_v<T_no_cv, unsigned long int>) {
        mpi_type = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T_no_cv, signed long long int>) {
        mpi_type = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T_no_cv, unsigned long long int>) {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T_no_cv, float>) {
        mpi_type = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T_no_cv, double>) {
        mpi_type = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T_no_cv, long double>) {
        mpi_type = MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T_no_cv, int8_t>) {
        mpi_type = MPI_INT8_T;
    } else if constexpr (std::is_same_v<T_no_cv, int16_t>) {
        mpi_type = MPI_INT16_T;
    } else if constexpr (std::is_same_v<T_no_cv, int32_t>) {
        mpi_type = MPI_INT32_T;
    } else if constexpr (std::is_same_v<T_no_cv, int64_t>) {
        mpi_type = MPI_INT64_T;
    } else if constexpr (std::is_same_v<T_no_cv, uint8_t>) {
        mpi_type = MPI_UINT8_T;
    } else if constexpr (std::is_same_v<T_no_cv, uint16_t>) {
        mpi_type = MPI_UINT16_T;
    } else if constexpr (std::is_same_v<T_no_cv, uint32_t>) {
        mpi_type = MPI_UINT32_T;
    } else if constexpr (std::is_same_v<T_no_cv, uint64_t>) {
        mpi_type = MPI_UINT64_T;
    } else if constexpr (std::is_same_v<T_no_cv, bool>) {
        mpi_type = MPI_C_BOOL;
    } else if constexpr (std::is_same_v<T_no_cv, std::complex<float>>) {
        mpi_type = MPI_C_FLOAT_COMPLEX;
    } else if constexpr (std::is_same_v<T_no_cv, std::complex<double>>) {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same_v<T_no_cv, std::complex<long double>>) {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    } else {
        mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    }

    KASSERT(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

/// @}

} // namespace kamping
