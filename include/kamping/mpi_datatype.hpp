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
#include <pfr.hpp>

#include "kamping/builtin_types.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/environment.hpp"
#include "kamping/noexcept.hpp"

namespace kamping {
struct kamping_tag {};
} // namespace kamping

namespace pfr {
template <typename T>
struct is_reflectable<T, kamping::kamping_tag> : std::true_type {};
} // namespace pfr

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{
///
///

namespace internal {
template <typename T>
struct is_std_pair : std::false_type {};
template <typename T1, typename T2>
struct is_std_pair<std::pair<T1, T2>> : std::true_type {};

template <typename T>
struct is_std_tuple : std::false_type {};
template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename A>
struct is_std_array : std::false_type {};
template <typename T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {
    using value_type             = T;
    static constexpr size_t size = N;
};
} // namespace internal

template <typename T, size_t N>
struct contiguous_type {
    static constexpr TypeCategory category            = TypeCategory::contiguous;
    static constexpr bool         has_to_be_committed = category_has_to_be_committed(category);
    static MPI_Datatype           data_type();
};

template <typename T>
struct byte_serialized : contiguous_type<std::byte, sizeof(T)> {};

template <typename T>
struct struct_type {
    static_assert(
        internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value
            || pfr::is_implicitly_reflectable<T, kamping_tag>::value,
        "Type must be a std::pair, std::tuple or reflectable"
    );
    static constexpr TypeCategory category            = TypeCategory::struct_like;
    static constexpr bool         has_to_be_committed = category_has_to_be_committed(category);
    static MPI_Datatype           data_type();
};

template <typename T>
auto select_type_trait() {
    using T_no_const = std::remove_const_t<T>;

    static_assert(
        !std::is_pointer_v<T_no_const>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?"
    );

    static_assert(!std::is_function_v<T_no_const>, "MPI does not support function types.");

    // TODO: this might be a bit too strict. We might want to allow unions in the future.
    static_assert(!std::is_union_v<T_no_const>, "MPI does not support union types.");

    static_assert(!std::is_void_v<T_no_const>, "There is no MPI datatype corresponding to void.");

    // map enum to underlying type
    if constexpr (is_builtin_type_v<T_no_const>) {
        return builtin_type<T_no_const>{};
    } else if constexpr (std::is_enum_v<T_no_const>) {
        return select_type_trait<std::underlying_type_t<T_no_const>>();
    } else if constexpr (std::is_array_v<T_no_const>) {
        constexpr size_t array_size = std::extent_v<T_no_const>;
        using underlying_type       = std::remove_extent_t<T_no_const>;
        return contiguous_type<underlying_type, array_size>{};
    } else if constexpr (internal::is_std_array<T_no_const>::value) {
        using underlying_type       = typename internal::is_std_array<T_no_const>::value_type;
        constexpr size_t array_size = internal::is_std_array<T_no_const>::size;
        return contiguous_type<underlying_type, array_size>{};
    } else if constexpr (std::is_trivially_copyable_v<T_no_const>) {
        return byte_serialized<T_no_const>{};
    } else {
        static_assert(
            // this should always evaluate to false
            !std::is_trivially_copyable_v<T_no_const>,
            "Type not supported directly by KaMPIng. Please provide a specialization for mpi_type_traits."
        );
    }
}

template <typename T>
struct mpi_type_traits : decltype(select_type_trait<T>()) {
    using base = decltype(select_type_trait<T>());
    /// @brief The category of the type.
    static constexpr TypeCategory category = base::category;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = category_has_to_be_committed(category);

    /// @brief The MPI_Datatype corresponding to the type T.
    static MPI_Datatype data_type() {
        return decltype(select_type_trait<T>())::data_type();
    }
};

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
template <typename T>
[[nodiscard]] MPI_Datatype mpi_datatype() KAMPING_NOEXCEPT {
    if constexpr (mpi_type_traits<T>::has_to_be_committed) {
        static MPI_Datatype type = construct_and_commit_type<T>();
        return type;
    } else {
        return mpi_type_traits<T>::data_type();
    }
}

template <typename T, size_t N>
MPI_Datatype contiguous_type<T, N>::data_type() {
    MPI_Datatype type;
    MPI_Type_contiguous(
        static_cast<int>(N),
        [] {
            if constexpr (std::is_same_v<T, std::byte>) {
                return MPI_BYTE;
            } else {
                return mpi_type_traits<T>::data_type();
            }
        }(),
        &type
    );
    return type;
}

namespace internal {
template <typename T, typename F, size_t... Is>
void for_each_tuple_field(T&& t, F&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(std::forward<T>(t)), Is), ...);
}
template <typename T, typename F>
void for_each_tuple_field(T& t, F&& f) {
    for_each_tuple_field(t, std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <typename T, typename F>
void for_each_field(T& t, F&& f) {
    if constexpr (internal::is_std_pair<T>::value || internal::is_std_tuple<T>::value) {
        for_each_tuple_field(t, std::forward<F>(f));
    } else {
        pfr::for_each_field(t, std::forward<F>(f));
    }
}

template <typename T>
constexpr size_t tuple_size = [] {
    if constexpr (internal::is_std_pair<T>::value) {
        return 2;
    } else if constexpr (internal::is_std_tuple<T>::value) {
        return std::tuple_size_v<T>;
    } else {
        return pfr::tuple_size_v<T>;
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
        types[i]     = mpi_type_traits<std::remove_reference_t<decltype(elem)>>::data_type();
        disp[i]      = MPI_Aint_diff(disp[i], base);
        blocklens[i] = 1;
    });
    MPI_Datatype type;
    int          err = MPI_Type_create_struct(static_cast<int>(internal::tuple_size<T>), blocklens, disp, types, &type);
    THROW_IF_MPI_ERROR(err, MPI_Type_create_struct);
    return type;
}

/// @}

} // namespace kamping
