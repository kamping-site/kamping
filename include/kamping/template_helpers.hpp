/// @file
/// @brief Template magic to implement named parameters in cpp

#pragma once

#include <cstddef>
#include <tuple>

#include "parameter_type_definitions.hpp"

namespace kamping {
namespace internal {

/// @addtogroup kamping_utility
/// @{


/// @brief returns position of first argument in Args with Trait trait and returns it
///
/// @tparam Trait trait with which an argument should be found
/// @tparam I index of current argument to evaluate
/// @tparam A argument to evaluate
/// @tparam Args all remaining arguments
/// @return position of first argument whit matched trait
///
template <ParameterType ptype, size_t I, typename Arg, typename... Args>
constexpr size_t find_pos() {
    if constexpr (Arg::ptype == ptype)
        return I;
    else
        return find_pos<ptype, I + 1, Args...>();
}
/// @brief returns parameter which the desired trait
///
/// @tparam Trait trait with which an argument should be found
/// @tparam Args all arguments to be searched
/// returns the first parameter whose type has the appropriate par_type
///
template <ParameterType ptype, typename... Args>
decltype(auto) select_ptype(Args&&... args) {
    return std::move(std::get<find_pos<ptype, 0, Args...>()>(std::forward_as_tuple(args...)));
}

// https://stackoverflow.com/a/9154394 TODO license?
template <typename>
struct true_type : std::true_type {};
template <typename T>
auto test_extract(int) -> true_type<decltype(std::declval<T>().extract())>;
template <typename T>
auto test_extract(...) -> std::false_type;
template <typename T>
struct has_extract : decltype(internal::test_extract<T>(0)) {};
template <typename T>
inline constexpr bool has_extract_v = has_extract<T>::value;

///@}
} // namespace internal
} // namespace kamping
