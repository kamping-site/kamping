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
/// @brief Mapping of C++ datatypes to builtin MPI types.

#pragma once
#include <complex>
#include <type_traits>

#include <mpi.h>

#include "kamping/kabool.hpp"

namespace kamping {

/// @addtogroup kamping_mpi_utility
/// @{

/// @brief the members specify which group the datatype belongs to according to the type groups specified in
/// Section 6.9.2 of the MPI 4.0 standard.
enum class TypeCategory { integer, floating, complex, logical, byte, character, struct_like, contiguous };

/// @brief Checks if a type of the given \p category has to commited before usage in MPI calls.
constexpr bool category_has_to_be_committed(TypeCategory category) {
    switch (category) {
        case TypeCategory::integer:
        case TypeCategory::floating:
        case TypeCategory::complex:
        case TypeCategory::logical:
        case TypeCategory::byte:
        case TypeCategory::character:
            return false;
        case TypeCategory::struct_like:
        case TypeCategory::contiguous:
            return true;
    }
}

/// @brief Checks if the type \p T is a builtin MPI type.
///
/// Provides a member constant \c value which is equal to \c true if \p T is a builtin type.
/// If `value` is `true`, the following members are defined, where \c data_type() returns the
/// corresponding \c MPI_Datatype and \c category the corresponding \ref TypeCategory.
///
/// ```cpp
///        struct builtin_type<T> {
///            static constexpr bool value = true;
///            static MPI_Datatype data_type();
///            static constexpr TypeCategory category;
///        };
/// ```
///
template <typename T>
struct builtin_type : std::false_type {};

/// @brief Helper variable template for \ref builtin_type.
template <typename T>
constexpr bool is_builtin_type_v = builtin_type<T>::value;

/// @brief Specialization of \ref builtin_type for \c char.
template <>
struct builtin_type<char> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::character; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `signed char`.
template <>
struct builtin_type<signed char> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_SIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `unsigned char`.
template <>
struct builtin_type<unsigned char> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `wchar_t`.
template <>
struct builtin_type<wchar_t> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_WCHAR;
    }
    static constexpr TypeCategory category = TypeCategory::character; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `short int`.
template <>
struct builtin_type<short int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `unsigned short int`.
template <>
struct builtin_type<unsigned short int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `int`.
template <>
struct builtin_type<int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_INT;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `unsigned int`.
template <>
struct builtin_type<unsigned int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `long int`.
template <>
struct builtin_type<long int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `unsigned long int`.
template <>
struct builtin_type<unsigned long int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `long long int`.
template <>
struct builtin_type<long long int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `unsigned long long int`.
template <>
struct builtin_type<unsigned long long int> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `float`.
template <>
struct builtin_type<float> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_FLOAT;
    }
    static constexpr TypeCategory category = TypeCategory::floating; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `double`.
template <>
struct builtin_type<double> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `long double`.
template <>
struct builtin_type<long double> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_LONG_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `bool`.
template <>
struct builtin_type<bool> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for \ref kabool.
template <>
struct builtin_type<kabool> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `std::complex<float>`.
template <>
struct builtin_type<std::complex<float>> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CXX_FLOAT_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `std::complex<double>`.
template <>
struct builtin_type<std::complex<double>> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex; ///< The types's \ref TypeCategory.
};

/// @brief Specialization of \ref builtin_type for `std::complex<long double>`.
template <>
struct builtin_type<std::complex<long double>> : std::true_type {
    /// @brief Returns the matching \c MPI_Datatype.
    static MPI_Datatype data_type() {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex; ///< The types's \ref TypeCategory.
};
/// @}
} // namespace kamping
