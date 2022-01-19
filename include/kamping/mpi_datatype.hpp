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

struct is_builtin_mpi_type_false {
    static constexpr bool is_builtin = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_float   = false;
    static constexpr bool is_complex = false;
    static constexpr bool is_logical = false;
    static constexpr bool is_byte    = false;
};

struct is_builtin_mpi_type_true : is_builtin_mpi_type_false {
    static constexpr bool is_builtin = true;
};

template <typename T>
struct mpi_type_traits_impl : is_builtin_mpi_type_false {};

template <>
struct mpi_type_traits_impl<char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CHAR;
    }
};

template <>
struct mpi_type_traits_impl<signed char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_SIGNED_CHAR;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<unsigned char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_CHAR;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<wchar_t> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_WCHAR;
    }
};

template <>
struct mpi_type_traits_impl<short int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_SHORT;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<unsigned short int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_SHORT;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_INT;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<unsigned int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<unsigned long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_LONG;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<unsigned long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG_LONG;
    }
    static constexpr bool is_integer = true;
};

template <>
struct mpi_type_traits_impl<float> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_FLOAT;
    }
    static constexpr bool is_float = true;
};

template <>
struct mpi_type_traits_impl<double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_DOUBLE;
    }
    static constexpr bool is_float = true;
};

template <>
struct mpi_type_traits_impl<long double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_DOUBLE;
    }
    static constexpr bool is_float = true;
};

template <>
struct mpi_type_traits_impl<bool> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr bool is_logical = true;
};
template <>
struct mpi_type_traits_impl<std::complex<float>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_FLOAT_COMPLEX;
    }
    static constexpr bool is_complex = true;
};
template <>
struct mpi_type_traits_impl<std::complex<double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
    static constexpr bool is_complex = true;
};

template <>
struct mpi_type_traits_impl<std::complex<long double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
    static constexpr bool is_complex = true;
};


template <typename T>
struct mpi_type_traits : mpi_type_traits_impl<std::remove_cv_t<T>> {};

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new custom continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// @tparam T The type to translate into a MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_datatype() noexcept {
    if constexpr (mpi_type_traits<T>::is_builtin) {
        return mpi_type_traits<T>::data_type();
    }

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
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if constexpr (std::is_enum_v<T_no_cv>) {
        return mpi_datatype<std::underlying_type_t<T_no_cv>>();
    } else {
        mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    }

    assert(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

/// @}

} // namespace kamping
