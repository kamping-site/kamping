#pragma once
#include <tuple>
#include <type_traits>

#include <mpi.h>

namespace kamping {

// /// @brief maps C++ types to builtin \c MPI_Datatypes
// ///
// /// the members specify which group the datatype belongs to according to the type groups specified in Section 5.9.2
// of
// /// the MPI 3.1 standard.
// /// @tparam T Type to map to an \c MPI_Datatype.
// template <typename T>
// struct mpi_type_traits {
//     /// @brief \c true, if the type maps to a builtin \c MPI_Datatype.
//     static constexpr bool is_builtin;
//     /// @brief Category the type belongs to according to the MPI standard.
//     static constexpr TypeCategory category;
//     /// @brief This member function is only available if \c is_builtin is true. If this is the case, it returns the
//     \c
//     /// MPI_Datatype
//     /// @returns Constant of type \c MPI_Datatype mapping to type \c T according the the MPI standard.
//     static MPI_Datatype data_type();
// };
/// @brief the members specify which group the datatype belongs to according to the type groups specified in
/// Section 5.9.2 of the MPI 3.1 standard.
enum class TypeCategory {
    integer,
    floating,
    complex,
    logical,
    byte,
    character,
    kamping_provided,
    user_provided,
    undefined
};

struct is_builtin_mpi_type_false {
    static constexpr bool         is_builtin  = false;
    static constexpr TypeCategory category    = TypeCategory::undefined;
    static MPI_Datatype           data_type() = delete;
};

/// @brief Base type for builtin types.
struct is_builtin_mpi_type_true : is_builtin_mpi_type_false {
    static constexpr bool is_builtin = true;
};

/// @brief Base template for implementation.
template <typename T, typename Enable = void>
struct mpi_type_traits : is_builtin_mpi_type_false {};

/// @brief wrapper for \c mpi_type_traits_impl which removes const qualifiers
template <typename T>
struct mpi_type_traits<T, std::enable_if_t<std::is_const_v<T>>> : mpi_type_traits<std::remove_const_t<T>> {};

template <typename T>
static constexpr bool has_static_type =
    mpi_type_traits<T>::is_builtin || mpi_type_traits<T>::category == TypeCategory::kamping_provided
    || mpi_type_traits<T>::category == TypeCategory::user_provided;

template <typename... Ts>
static constexpr bool all_have_static_types = sizeof...(Ts) > 0 && (has_static_type<Ts> && ...);

template <typename T>
static constexpr bool tuple_all_have_static_types = false;

template <typename... Ts>
static constexpr bool tuple_all_have_static_types<std::tuple<Ts...>> = all_have_static_types<Ts...>;

struct kamping_tag {};
} // namespace kamping
