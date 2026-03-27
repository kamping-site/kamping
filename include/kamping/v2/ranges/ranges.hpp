#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "kamping/builtin_types.hpp"

namespace kamping::ranges {

template <class T>
concept integer_like = std::integral<T> && !std::same_as<T, bool>;

template <class T>
concept ptr_to_object = std::is_pointer_v<T> && std::is_object_v<std::remove_pointer_t<T>>;

template <typename>
inline constexpr bool mpi_enabled = false;

template <typename T>
concept is_mpi_enabled = kamping::ranges::mpi_enabled<std::remove_cvref_t<T>>;

template <typename T>
concept has_mpi_compatible_size_member = is_mpi_enabled<T> && requires(std::remove_reference_t<T> const& t) {
    { t.mpi_size() } -> integer_like;
};

template <typename T>
concept has_mpi_compatible_data_member = is_mpi_enabled<T> && requires(T&& t) {
    { t.mpi_data() } -> ptr_to_object;
};

template <typename T>
concept has_mpi_compatible_type_member = is_mpi_enabled<T> && requires(std::remove_reference_t<T> const& t) {
    { t.mpi_type() } -> std::convertible_to<MPI_Datatype>;
};
  
template <typename T>
concept range_of_builtin_mpi_type =
    std::ranges::range<T> && kamping::is_builtin_type_v<std::remove_cvref_t<std::ranges::range_value_t<T>>>;

template <has_mpi_compatible_size_member T>
constexpr auto size(T&& t) {
    return t.mpi_size();
}

template <std::ranges::sized_range T>
    requires(!has_mpi_compatible_size_member<T>)
constexpr auto size(T&& t) {
    return std::ranges::size(std::forward<T>(t));
}

template <has_mpi_compatible_data_member T>
constexpr auto data(T&& t) {
    return t.mpi_data();
}

template <std::ranges::contiguous_range T>
    requires(!has_mpi_compatible_data_member<T>)
constexpr auto data(T&& t) {
    return std::ranges::data(std::forward<T>(t));
}

template <has_mpi_compatible_type_member T>
constexpr auto type(T&& t) {
    return t.mpi_type();
}

template <range_of_builtin_mpi_type T>
    requires(!has_mpi_compatible_type_member<T>)
constexpr auto type(T&& /* t */) {
    return builtin_type<std::remove_cvref_t<std::ranges::range_value_t<T>>>::data_type();
}

} // namespace kamping::ranges
