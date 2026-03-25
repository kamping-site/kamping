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
/// @brief Struct-like MPI type construction via field reflection.

#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#ifdef KAMPING_ENABLE_REFLECTION
    #include <boost/pfr.hpp>
#endif

#include "kamping/types/builtin_types.hpp"
#include "kamping/types/detail/type_helpers.hpp"
#include "kamping/types/mpi_type_traits.hpp"

namespace kamping {

/// @addtogroup kamping_types
/// @{

/// @brief Tag used for indicating that a struct is reflectable.
/// @see struct_type
struct kamping_tag {};

/// @brief Constructs an MPI_Datatype for a struct-like type.
/// @tparam T The type to construct the MPI_Datatype for.
///
/// This requires that \p T is a `std::pair`, `std::tuple` or a type that is reflectable with
/// [pfr](https://github.com/boostorg/pfr). If you do not agree with PFR's decision if a type is implicitly
/// reflectable, you can override it by providing a specialization of \c pfr::is_reflectable with the tag \ref
/// kamping_tag.
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

/// @}

namespace internal {

/// @brief Applies functor \p f to each field of the tuple with an index in index sequence \p Is.
template <typename T, typename F, size_t... Is>
void for_each_tuple_field(T&& t, F&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(std::forward<T>(t)), Is), ...);
}

/// @brief Applies functor \p f to each field of the tuple \p t.
template <typename T, typename F>
void for_each_tuple_field(T& t, F&& f) {
    for_each_tuple_field(t, std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<T>>{});
}

/// @brief Applies functor \p f to each field of the tuple-like type \p t.
/// Works for `std::pair` and `std::tuple`. If KaMPIng's reflection support is enabled, also works
/// with types reflectable via [pfr](https://github.com/boostorg/pfr).
template <typename T, typename F>
void for_each_field(T& t, F&& f) {
    if constexpr (internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value) {
        for_each_tuple_field(t, std::forward<F>(f));
    } else {
#ifdef KAMPING_ENABLE_REFLECTION
        boost::pfr::for_each_field(t, std::forward<F>(f));
#else
        static_assert(internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value);
#endif
    }
}

/// @brief The number of elements in a tuple-like type.
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
    KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_create_struct failed");
    MPI_Datatype resized_type;
    err = MPI_Type_create_resized(type, 0, sizeof(T), &resized_type);
    KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_create_resized failed");
    return resized_type;
}

} // namespace kamping
