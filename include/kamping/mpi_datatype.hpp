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

#include "kamping/checking_casts.hpp"
#include "kamping/environment.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/noexcept.hpp"
#include "pfr.hpp"

namespace kamping::internal {

/// @brief Construct a custom continuous MPI datatype.
///
/// @param num_bytes_unsigned The number of bytes for the new type.
/// @return The newly created MPI_Datatype.
/// @see mpi_datatype()
///
inline MPI_Datatype construct_custom_continuous_type(size_t const num_bytes_unsigned) {
    int const    num_bytes = asserting_cast<int>(num_bytes_unsigned);
    MPI_Datatype type      = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(num_bytes, MPI_CHAR, &type);
    std::cout << "Committing custom continuous type of " << num_bytes << " bytes." << std::endl;
    MPI_Type_commit(&type);
    KASSERT(type != MPI_DATATYPE_NULL);
    mpi_env.register_mpi_type(type);
    return type;
}
} // namespace kamping::internal

namespace kamping {
/// @brief Wrapper around bool to allow handling containers of boolean values
class kabool {
public:
    /// @brief default constructor for a \c kabool with value \c false
    constexpr kabool() noexcept : _value() {}
    /// @brief constructor to construct a \c kabool out of a \c bool
    constexpr kabool(bool value) noexcept : _value(value) {}

    /// @brief implicit cast of \c kabool to \c bool
    inline constexpr operator bool() const noexcept {
        return _value;
    }

private:
    bool _value; /// < the wrapped boolean value
};

/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Creates a custom continuous MPI datatype.
///
/// @tparam NumBytes The number of bytes for the new type.
/// @return The newly created MPI_Datatype.
/// @see mpi_datatype()
///
template <size_t NumBytes>
[[nodiscard]] MPI_Datatype mpi_custom_continuous_type() KAMPING_NOEXCEPT {
    static_assert(NumBytes > 0, "You cannot create a continuous type with 0 bytes.");
    // Create a new MPI datatype only the first type per NumBytes this function is called.
    // By initializing this in the same line as the static declaration, this is thread safe.
    static MPI_Datatype type = internal::construct_custom_continuous_type(NumBytes);
    // From the second call onwards, re-use the existing type.
    return type;
}

/// @brief the members specify which group the datatype belongs to according to the type groups specified in
/// Section 5.9.2 of the MPI 3.1 standard.
enum class TypeCategory { integer, floating, complex, logical, byte, kamping_provided, user_provided, undefined };

#ifdef KAMPING_DOXYGEN_ONLY
/// @brief maps C++ types to builtin \c MPI_Datatypes
///
/// the members specify which group the datatype belongs to according to the type groups specified in Section 5.9.2 of
/// the MPI 3.1 standard.
/// @tparam T Type to map to an \c MPI_Datatype.
template <typename T>
struct mpi_type_traits {
    /// @brief \c true, if the type maps to a builtin \c MPI_Datatype.
    static constexpr bool is_builtin;
    /// @brief Category the type belongs to according to the MPI standard.
    static constexpr TypeCategory category;
    /// @brief This member function is only available if \c is_builtin is true. If this is the case, it returns the \c
    /// MPI_Datatype
    /// @returns Constant of type \c MPI_Datatype mapping to type \c T according the the MPI standard.
    static MPI_Datatype data_type();
};
#else
/// @brief Base type for non-builtin types.
struct is_builtin_mpi_type_false {
    static constexpr bool         is_builtin = false;
    static constexpr TypeCategory category   = TypeCategory::undefined;
};

/// @brief Base type for builtin types.
struct is_builtin_mpi_type_true : is_builtin_mpi_type_false {
    static constexpr bool is_builtin = true;
};

/// @brief Base template for implementation.
template <typename T>
struct mpi_type_traits_impl : is_builtin_mpi_type_false {};

// template specializations of mpi_type_traits_impl

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
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<unsigned char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
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
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<unsigned short int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_INT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<unsigned int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<unsigned long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<unsigned long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits_impl<float> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_FLOAT;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits_impl<double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits_impl<long double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits_impl<bool> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct mpi_type_traits_impl<kabool> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct mpi_type_traits_impl<std::complex<float>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_FLOAT_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};
template <>
struct mpi_type_traits_impl<std::complex<double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

template <>
struct mpi_type_traits_impl<std::complex<long double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

/// @brief wrapper for \c mpi_type_traits_impl which removes const and volatile qualifiers
template <typename T, typename Enable = void>
struct mpi_type_traits : mpi_type_traits_impl<std::remove_cv_t<T>> {};
#endif

template <typename T>
inline MPI_Datatype construct_and_commit_type() {
    MPI_Datatype type = mpi_type_traits<T>::data_type();
    std::cout << "Committing custom static type of " << typeid(T).name() << std::endl;
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
    if constexpr (mpi_type_traits<T>::is_builtin) {
        return mpi_type_traits<T>::data_type();
    } else if constexpr (mpi_type_traits<T>::category == TypeCategory::kamping_provided || mpi_type_traits<T>::category == TypeCategory::user_provided) {
        static MPI_Datatype type = construct_and_commit_type<T>();
        return type;
    }

    // Remove const and volatile qualifiers.
    // TODO: it is not clear if we should remove volatile here, because MPI does not support volatile pointers and
    // removing volatile from a pointer is undefined behavior.
    using T_no_cv = std::remove_cv_t<T>;

    // Check if we got a pointer type -> error
    static_assert(
        !std::is_pointer_v<T_no_cv>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?"
    );

    // Check if we got a function type -> error
    static_assert(!std::is_function_v<T_no_cv>, "MPI does not support function types.");

    // Check if we got a union type -> error
    static_assert(!std::is_union_v<T_no_cv>, "MPI does not support union types.");

    // Check if we got void -> error
    static_assert(!std::is_void_v<T_no_cv>, "There is no MPI datatype corresponding to void.");

    // Check if we got an array type -> create a continuous type.
    if constexpr (std::is_array_v<T_no_cv>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        return mpi_custom_continuous_type<sizeof(T_no_cv)>();
    }

    // Check if we got an enum type -> use underlying type
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    if constexpr (std::is_enum_v<T_no_cv>) {
        return mpi_datatype<std::underlying_type_t<T_no_cv>>();
    } else {
        mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    }

    KASSERT(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

/// @brief Gets the size of an MPI datatype in bytes.
///
/// @param mpi_datatype The MPI datatype to get the size of.
/// @return The size of the MPI datatype in bytes.
///
inline int mpi_datatype_size(MPI_Datatype mpi_datatype) {
    int                  result;
    [[maybe_unused]] int err = MPI_Type_size(mpi_datatype, &result);
    THROW_IF_MPI_ERROR(err, MPI_Type_size);
    return result;
}

template <typename T>
static constexpr bool has_static_type =
    mpi_type_traits<T>::is_builtin || mpi_type_traits<T>::category == TypeCategory::kamping_provided
    || mpi_type_traits<T>::category == TypeCategory::user_provided;

template <typename T1, typename T2>
struct mpi_type_traits<std::pair<T1, T2>, std::enable_if_t<has_static_type<T1> && has_static_type<T2>>>
    : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;
    static MPI_Datatype           data_type() {
        std::pair<T1, T2> t;
        MPI_Datatype      types[2]     = {mpi_datatype<T1>(), mpi_datatype<T2>()};
        int               blocklens[2] = {1, 1};
        MPI_Aint          base;
        MPI_Get_address(&t, &base);
        MPI_Aint disp[2];
        MPI_Get_address(&t.first, &disp[0]);
        MPI_Get_address(&t.second, &disp[1]);
        disp[0] = MPI_Aint_diff(disp[0], base);
        disp[1] = MPI_Aint_diff(disp[1], base);
        MPI_Datatype type;
        MPI_Type_create_struct(2, blocklens, disp, types, &type);
        return type;
    }
};

template <typename... Ts>
struct mpi_type_traits<std::tuple<Ts...>, std::enable_if_t<(sizeof...(Ts) > 0 && (has_static_type<Ts> && ...))>>
    : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;
    static MPI_Datatype           data_type() {
        std::tuple<Ts...>     t;
        constexpr std::size_t tuple_size = sizeof...(Ts);

        MPI_Datatype types[tuple_size] = {mpi_datatype<Ts>()...};
        int          blocklens[tuple_size];
        MPI_Aint     disp[tuple_size];
        MPI_Aint     base;
        MPI_Get_address(&t, &base);

        // Calculate displacements for each tuple element using std::apply and fold expressions
        size_t i = 0;
        std::apply(
            [&](auto&... elem) {
                (
                    [&] {
                        MPI_Get_address(&elem, &disp[i]);
                        disp[i]      = MPI_Aint_diff(disp[i], base);
                        blocklens[i] = 1;
                        i++;
                    }(),
                    ...
                );
            },
            t
        );

        MPI_Datatype type;
        MPI_Type_create_struct(tuple_size, blocklens, disp, types, &type);
        return type;
    }
};

template <typename E>
struct mpi_type_traits<E, std::enable_if_t<std::is_enum_v<E> && has_static_type<std::underlying_type_t<E>>>>
    : mpi_type_traits<std::underlying_type_t<E>> {};

struct kamping_tag {};

template <typename T>
using member_types = decltype(pfr::structure_to_tuple(std::declval<T>()));

// TODO: only enable if each member has a static type
template <typename T>
struct mpi_type_traits<T, std::enable_if_t<pfr::is_implicitly_reflectable<T, kamping_tag>::value>>
    : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;
    static MPI_Datatype           data_type() {
        T                     t;
        constexpr std::size_t tuple_size = pfr::tuple_size_v<T>;

        MPI_Datatype types[tuple_size];
        int          blocklens[tuple_size];
        MPI_Aint     disp[tuple_size];
        MPI_Aint     base;
        MPI_Get_address(&t, &base);

        // Calculate displacements for each tuple element using std::apply and fold expressions
        pfr::for_each_field(t, [&](auto&& elem, size_t i) {
            MPI_Get_address(&elem, &disp[i]);
            types[i]     = mpi_datatype<std::remove_cv_t<std::remove_reference_t<decltype(elem)>>>();
            disp[i]      = MPI_Aint_diff(disp[i], base);
            blocklens[i] = 1;
        });

        MPI_Datatype type;
        MPI_Type_create_struct(tuple_size, blocklens, disp, types, &type);
        return type;
    }
};

/// @}

} // namespace kamping
