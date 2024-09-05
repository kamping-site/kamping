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

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>
#ifdef KAMPING_ENABLE_REFLECTION
    #include <boost/pfr.hpp>
#endif

#include "kamping/builtin_types.hpp"
#include "kamping/environment.hpp"
#include "kamping/noexcept.hpp"

namespace kamping {
/// @brief Tag used for indicating that a struct is reflectable.
/// @see struct_type
struct kamping_tag {};
} // namespace kamping

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{
///
///

namespace internal {
/// @brief Helper to check if a type is a `std::pair`.
template <typename T>
struct is_std_pair : std::false_type {};
/// @brief Helper to check if a type is a `std::pair`.
template <typename T1, typename T2>
struct is_std_pair<std::pair<T1, T2>> : std::true_type {};

/// @brief Helper to check if a type is a `std::tuple`.
template <typename T>
struct is_std_tuple : std::false_type {};
/// @brief Helper to check if a type is a `std::tuple`.
template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

/// @brief Helper to check if a type is a `std::array`.
template <typename A>
struct is_std_array : std::false_type {};

/// @brief Helper to check if a type is a `std::array`.
template <typename T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {
    using value_type             = T; ///< The type of the elements in the array.
    static constexpr size_t size = N; ///< The number of elements in the array.
};
} // namespace internal

/// @brief Constructs an contiguous type of size \p N from type \p T using `MPI_Type_contiguous`.
template <typename T, size_t N>
struct contiguous_type {
    static constexpr TypeCategory category = TypeCategory::contiguous; ///< The category of the type.
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = category_has_to_be_committed(category);
    /// @brief The MPI_Datatype corresponding to the type.
    static MPI_Datatype data_type();
};

/// @brief Constructs a type that is serialized as a sequence of `sizeof(T)` bytes using `MPI_BYTE`. Note that you must
/// ensure that this conversion is valid.
template <typename T>
struct byte_serialized : contiguous_type<std::byte, sizeof(T)> {};

/// @brief Constructs a MPI_Datatype for a struct-like type.
/// @tparam T The type to construct the MPI_Datatype for.
///
/// This requires that \p T is a `std::pair`, `std::tuple` or a type that is reflectable with
/// [pfr](https://github.com/boostorg/pfr). If you do not agree with PFR's decision if a type is implicitly
/// reflectable, you can override it by providing a specialization of \c pfr::is_reflectable with the tag \ref
/// kamping_tag.
/// @see https://apolukhin.github.io/pfr_non_boost/pfr/is_reflectable.html
/// https://www.boost.org/doc/libs/master/doc/html/reference_section_of_pfr.htmlfor details
/// @note Reflection support for arbitrary struct types is only suppported when KaMPIng is compiled with PFR.
template <typename T>
struct struct_type {
#ifdef KAMPING_ENABLE_REFLECTION
    static_assert(
        internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value
            || boost::pfr::is_implicitly_reflectable<T, kamping_tag>::value,
        "Type must be a std::pair, std::tuple or reflectable"
    );
#else
    static_assert(
        internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value, "Type must be a std::pair or std::tuple"
    );
#endif
    /// @brief The category of the type.
    static constexpr TypeCategory category = TypeCategory::struct_like;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = category_has_to_be_committed(category);
    /// @brief The MPI_Datatype corresponding to the type.
    static MPI_Datatype data_type();
};

/// @brief Type tag for indicating that no static type definition exists for a type.
struct no_matching_type {};

/// @brief The type dispatcher that maps a C++ type \p T to a type trait that can be used to construct an MPI_Datatype.
///
/// The mapping is as follows:
/// - C++ types directly supported by MPI are mapped to the corresponding `MPI_Datatype`.
/// - Enums are mapped to the underlying type.
/// - C-style arrays and `std::array` are mapped to contiguous types of the underlying type.
/// - All other trivially copyable types are mapped to a contiguous type consisting of `sizeof(T)` bytes.
/// - All other types are not supported directly and require a specialization of `mpi_type_traits`. In this case, the
///  trait `no_matching_type` is returned.
///
/// @returns The corresponding type trait for the type \p T.
template <typename T>
auto type_dispatcher() {
    using T_no_const = std::remove_const_t<T>; // remove const from T
                                               // we previously also removed volatile here, but interpreting a pointer
                                               // to a volatile type as a pointer to a non-volatile type is UB

    static_assert(
        !std::is_pointer_v<T_no_const>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?"
    );

    static_assert(!std::is_function_v<T_no_const>, "MPI does not support function types.");

    // TODO: this might be a bit too strict. We might want to allow unions in the future.
    static_assert(!std::is_union_v<T_no_const>, "MPI does not support union types.");

    static_assert(!std::is_void_v<T_no_const>, "There is no MPI datatype corresponding to void.");

    if constexpr (is_builtin_type_v<T_no_const>) {
        // builtin types are handled by the builtin_type trait
        return builtin_type<T_no_const>{};
    } else if constexpr (std::is_enum_v<T_no_const>) {
        // enums are mapped to the underlying type
        return type_dispatcher<std::underlying_type_t<T_no_const>>();
    } else if constexpr (std::is_array_v<T_no_const>) {
        // arrays are mapped to contiguous types
        constexpr size_t array_size = std::extent_v<T_no_const>;
        using underlying_type       = std::remove_extent_t<T_no_const>;
        return contiguous_type<underlying_type, array_size>{};
    } else if constexpr (internal::is_std_array<T_no_const>::value) {
        // std::array is mapped to contiguous types
        using underlying_type       = typename internal::is_std_array<T_no_const>::value_type;
        constexpr size_t array_size = internal::is_std_array<T_no_const>::size;
        return contiguous_type<underlying_type, array_size>{};
    } else if constexpr (std::is_trivially_copyable_v<T_no_const>) {
        // all other trivially copyable types are mapped to a sequence of bytes
        return byte_serialized<T_no_const>{};
    } else {
        return no_matching_type{};
    }
}

/// @brief The type trait that maps a C++ type \p T to a type trait that can be used to construct an MPI_Datatype.
///
/// The default behavior is controlled by \ref type_dispatcher. If you want to support a type that is not supported by
/// the default behavior, you can specialize this trait. For example:
///
/// ```cpp
/// struct MyType {
///    int a;
///    double b;
///    char c;
///    std::array<int, 3> d;
/// };
/// namespace kamping {
/// // using KaMPIng's built-in struct serializer
/// template <>
/// struct mpi_type_traits<MyType> : struct_type<MyType> {};
///
/// // or using an explicitly constructed type
/// template <>
/// struct mpi_type_traits<MyType> {
///    static constexpr bool has_to_be_committed = true;
///    static MPI_Datatype data_type() {
///        MPI_Datatype type;
///        MPI_Type_create_*(..., &type);
///        return type;
///    }
/// };
/// } // namespace kamping
/// ```
///
template <typename T, typename Enable = void>
struct mpi_type_traits {};

/// @brief The type trait that maps a C++ type \p T to a type trait that can be used to construct an MPI_Datatype.
///
/// The default behavior is controlled by \ref type_dispatcher. If you want to support a type that is not supported by
/// the default behavior, you can specialize this trait. For example:
///
/// ```cpp
/// struct MyType {
///    int a;
///    double b;
///    char c;
///    std::array<int, 3> d;
/// };
/// namespace kamping {
/// // using KaMPIng's built-in struct serializer
/// template <>
/// struct mpi_type_traits<MyType> : struct_type<MyType> {};
///
/// // or using an explicitly constructed type
/// template <>
/// struct mpi_type_traits<MyType> {
///    static constexpr bool has_to_be_committed = true;
///    static MPI_Datatype data_type() {
///        MPI_Datatype type;
///        MPI_Type_create_*(..., &type);
///        return type;
///    }
/// };
/// } // namespace kamping
/// ```
///
template <typename T>
struct mpi_type_traits<T, std::enable_if_t<!std::is_same_v<decltype(type_dispatcher<T>()), no_matching_type>>> {
    /// @brief The base type of this trait obtained via \ref type_dispatcher.
    /// This defines how the data type is constructed in \c mpi_type_traits::data_type().
    using base = decltype(type_dispatcher<T>());
    /// @brief The category of the type.
    static constexpr TypeCategory category = base::category;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = category_has_to_be_committed(category);
    /// @brief The MPI_Datatype corresponding to the type T.
    static MPI_Datatype data_type() {
        return decltype(type_dispatcher<T>())::data_type();
    }
};

///@brief Check if the type has a static type definition, i.e. \ref mpi_type_traits is defined.
template <typename, typename Enable = void>
struct has_static_type : std::false_type {};

///@brief Check if the type has a static type definition, i.e. \ref mpi_type_traits is defined.
template <typename T>
struct has_static_type<T, std::void_t<decltype(mpi_type_traits<T>::data_type())>> : std::true_type {};

///@brief Check if the type has a static type definition, i.e. has a corresponding \c MPI_Datatype defined following the
/// rules from \ref type_dispatcher.
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
    KASSERT(type != MPI_DATATYPE_NULL);
    mpi_env.register_mpi_type(type);
    return type;
}

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new  continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// To check if type \c T maps to a builtin \c MPI_Datatype at compile-time, use \c mpi_type_traits.
/// @tparam T The type to translate into an MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///

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

template <typename T, size_t N>
MPI_Datatype contiguous_type<T, N>::data_type() {
    MPI_Datatype type;
    MPI_Datatype base_type;
    if constexpr (std::is_same_v<T, std::byte>) {
        base_type = MPI_BYTE;
    } else {
        static_assert(
            has_static_type_v<T>,
            "\n --> Type not supported directly by KaMPIng. Please provide a specialization for mpi_type_traits."
        );
        base_type = mpi_type_traits<T>::data_type();
    }
    int count = static_cast<int>(N);
    int err   = MPI_Type_contiguous(count, base_type, &type);
    THROW_IF_MPI_ERROR(err, MPI_Type_contiguous);
    return type;
}

namespace internal {
/// @brief Applies functor \p f to each field of the tuple with an index in index sequence \p Is.
///
/// \p f should be a callable that takes a reference to the field and its index.
template <typename T, typename F, size_t... Is>
void for_each_tuple_field(T&& t, F&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(std::forward<T>(t)), Is), ...);
}

/// @brief Applies functor \p f to each field of the tuple \p t.
///
/// \p f should be a callable that takes a reference to the field and its index.
template <typename T, typename F>
void for_each_tuple_field(T& t, F&& f) {
    for_each_tuple_field(t, std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<T>>{});
}

/// @brief Applies functor \p f to each field of the tuple-like type \p t.
/// This works for `std::pair` and `std::tuple`. If KaMPIng's reflection support is enabled, this also works with types
/// that are reflectable with [pfr](https://github.com/boostorg/pfr).
///
/// \p f should be a callable that takes a reference to the field and
/// its index.
template <typename T, typename F>
void for_each_field(T& t, F&& f) {
    if constexpr (internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value) {
        for_each_tuple_field(t, std::forward<F>(f));
    } else {
#ifdef KAMPING_ENABLE_REFLECTION
        boost::pfr::for_each_field(t, std::forward<F>(f));
#else
        // should not happen
        static_assert(internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value);
#endif
    }
}

/// @brief The number of elements in a tuple-like type.
/// This works for `std::pair` and `std::tuple`.
/// If KaMPIng's reflection support is enabled, this also works with types that are reflectable with
/// [pfr](https://github.com/boostorg/pfr).
template <typename T>
constexpr size_t tuple_size = [] {
    if constexpr (internal::is_std_pair<T>::value) {
        return 2;
    } else if constexpr (internal::is_std_tuple<T>::value) {
        return std::tuple_size_v<T>;
    } else {
#ifdef KAMPING_ENABLE_REFLECTION
        return boost::pfr::tuple_size_v<T>;
#else
        if constexpr (std::is_arithmetic_v<T>) {
            return 1;
        } else {
            return std::tuple_size_v<T>;
        }
#endif
    }
}();
} // namespace internal

template <typename T>
MPI_Datatype struct_type<T>::data_type() {
    T        t{};
    MPI_Aint base;
    MPI_Get_address(&t, &base);
    int          blocklens[internal::tuple_size<T>];
    MPI_Datatype types[internal::tuple_size<T>];
    MPI_Aint     disp[internal::tuple_size<T>];
    internal::for_each_field(t, [&](auto& elem, size_t i) {
        MPI_Get_address(&elem, &disp[i]);
        using elem_type = std::remove_reference_t<decltype(elem)>;
        static_assert(
            has_static_type_v<elem_type>,
            "\n --> Type not supported directly by KaMPIng. Please provide a specialization for mpi_type_traits."
        );
        types[i]     = mpi_type_traits<elem_type>::data_type();
        disp[i]      = MPI_Aint_diff(disp[i], base);
        blocklens[i] = 1;
    });
    MPI_Datatype type;
    int          err = MPI_Type_create_struct(static_cast<int>(internal::tuple_size<T>), blocklens, disp, types, &type);
    THROW_IF_MPI_ERROR(err, MPI_Type_create_struct);
    MPI_Datatype resized_type;
    err = MPI_Type_create_resized(type, 0, sizeof(T), &resized_type);
    THROW_IF_MPI_ERROR(err, MPI_Type_create_resized);
    return resized_type;
}

/// @brief A scoped MPI_Datatype that commits the type on construction and frees it on destruction.
/// This is useful for RAII-style management of MPI_Datatype objects.
class ScopedDatatype {
    MPI_Datatype _type; ///< The MPI_Datatype.
public:
    /// @brief Construct a new scoped MPI_Datatype and commits it. If no type is provided, default to
    /// `MPI_DATATYPE_NULL` and does not commit or free anything.
    ScopedDatatype(MPI_Datatype type = MPI_DATATYPE_NULL) : _type(type) {
        if (type != MPI_DATATYPE_NULL) {
            mpi_env.commit(type);
        }
    }
    /// @brief Deleted copy constructor.
    ScopedDatatype(ScopedDatatype const&) = delete;
    /// @brief Deleted copy assignment.
    ScopedDatatype& operator=(ScopedDatatype const&) = delete;

    /// @brief Move constructor.
    ScopedDatatype(ScopedDatatype&& other) noexcept : _type(other._type) {
        other._type = MPI_DATATYPE_NULL;
    }
    /// @brief Move assignment.
    ScopedDatatype& operator=(ScopedDatatype&& other) noexcept {
        std::swap(_type, other._type);
        return *this;
    }
    /// @brief Get the MPI_Datatype.
    MPI_Datatype const& data_type() const {
        return _type;
    }
    /// @brief Free the MPI_Datatype.
    ~ScopedDatatype() {
        if (_type != MPI_DATATYPE_NULL) {
            mpi_env.free(_type);
        }
    }
};

/// @}

} // namespace kamping
