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


#pragma once

#include <cassert>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <unordered_set>

#include <mpi.h>

namespace kamping {

/// @addtogroup kamping_mpi_utility
/// @{
///
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

enum class DatatypeCategory { integer, floating, complex, logical, byte, custom, none };

struct DatatypeInfo {
    MPI_Datatype     type     = MPI_DATATYPE_NULL;
    DatatypeCategory category = DatatypeCategory::custom;
};

/// @brief Translate template parameter T to a a builtin MPI_Datatype. If no corresponding MPI_Datatype exists,
/// `MPI_DATATYPE_NULL` is returned.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// @tparam T The type to translate into a MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or `MPI_DATATYPE_NULL`
///
template <typename T>
[[nodiscard]] constexpr DatatypeInfo get_mpi_datatype_info() noexcept {
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
    if (std::is_array_v<T_no_cv>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        /// @todo this seems to be wrong
        return {mpi_custom_continuous_type<sizeof(T_no_cv)>(), DatatypeCategory::custom};
    }

    // Check if we got a enum type -> use underlying type
    if constexpr (std::is_enum_v<T_no_cv>) {
        return get_mpi_datatype_info<std::underlying_type_t<T_no_cv>>();
    }

    DatatypeInfo type_info;
    type_info.type = MPI_DATATYPE_NULL;
    if (std::is_same_v<T_no_cv, char>) {
        type_info.type     = MPI_CHAR;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, signed char>) {
        type_info.type     = MPI_SIGNED_CHAR;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, unsigned char>) {
        type_info.type     = MPI_UNSIGNED_CHAR;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, wchar_t>) {
        type_info.type     = MPI_WCHAR;
        type_info.category = DatatypeCategory::none;
    } else if (std::is_same_v<T_no_cv, signed short>) {
        type_info.type     = MPI_SHORT;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, unsigned short>) {
        type_info.type     = MPI_UNSIGNED_SHORT;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, signed int>) {
        type_info.type     = MPI_INT;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, unsigned int>) {
        type_info.type     = MPI_UNSIGNED;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, signed long int>) {
        type_info.type     = MPI_LONG;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, unsigned long int>) {
        type_info.type     = MPI_UNSIGNED_LONG;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, signed long long int>) {
        type_info.type     = MPI_LONG_LONG;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, unsigned long long int>) {
        type_info.type     = MPI_UNSIGNED_LONG_LONG;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, float>) {
        type_info.type     = MPI_FLOAT;
        type_info.category = DatatypeCategory::floating;
    } else if (std::is_same_v<T_no_cv, double>) {
        type_info.type     = MPI_DOUBLE;
        type_info.category = DatatypeCategory::floating;
    } else if (std::is_same_v<T_no_cv, long double>) {
        type_info.type     = MPI_LONG_DOUBLE;
        type_info.category = DatatypeCategory::floating;
    } else if (std::is_same_v<T_no_cv, int8_t>) {
        type_info.type     = MPI_INT8_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, int16_t>) {
        type_info.type     = MPI_INT16_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, int32_t>) {
        type_info.type     = MPI_INT32_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, int64_t>) {
        type_info.type     = MPI_INT64_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, uint8_t>) {
        type_info.type     = MPI_UINT8_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, uint16_t>) {
        type_info.type     = MPI_UINT16_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, uint32_t>) {
        type_info.type     = MPI_UINT32_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, uint64_t>) {
        type_info.type     = MPI_UINT64_T;
        type_info.category = DatatypeCategory::integer;
    } else if (std::is_same_v<T_no_cv, bool>) {
        type_info.type     = MPI_CXX_BOOL;
        type_info.category = DatatypeCategory::logical;
    } else if (std::is_same_v<T_no_cv, std::complex<float>>) {
        type_info.type     = MPI_CXX_FLOAT_COMPLEX;
        type_info.category = DatatypeCategory::complex;
    } else if (std::is_same_v<T_no_cv, std::complex<double>>) {
        type_info.type     = MPI_CXX_DOUBLE_COMPLEX;
        type_info.category = DatatypeCategory::complex;
    } else if (std::is_same_v<T_no_cv, std::complex<long double>>) {
        type_info.type     = MPI_CXX_LONG_DOUBLE_COMPLEX;
        type_info.category = DatatypeCategory::complex;
    }
    return type_info;
}

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new custom continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// @tparam T The type to translate into a MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///
template <typename T>
[[nodiscard]] MPI_Datatype mpi_datatype() noexcept {
    DatatypeInfo mpi_type = get_mpi_datatype_info<T>();
    if (mpi_type.type == MPI_DATATYPE_NULL) {
        mpi_type.type = mpi_custom_continuous_type<sizeof(T)>();
    }

    return mpi_type.type;
}
template <typename T>
constexpr bool is_mpi_integer() noexcept {
    auto type = get_mpi_datatype_info<T>();
    return type.category == DatatypeCategory::integer;
}

template <typename T>
constexpr bool is_mpi_float() noexcept {
    auto type = get_mpi_datatype_info<T>();
    return type.category == DatatypeCategory::floating;
}

template <typename T>
constexpr bool is_mpi_logical() noexcept {
    auto type = get_mpi_datatype_info<T>();
    return type.category == DatatypeCategory::logical;
}

template <typename T>
constexpr bool is_mpi_complex() noexcept {
    auto type = get_mpi_datatype_info<T>();
    return type.category == DatatypeCategory::complex;
}

template <typename T>
constexpr bool is_mpi_byte() noexcept {
    auto type = get_mpi_datatype_info<T>();
    return type.category == DatatypeCategory::byte;
}

/// @}

} // namespace kamping
