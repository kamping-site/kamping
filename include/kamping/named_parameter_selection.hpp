// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:

/// @file
/// @brief Template magic to implement named parameters in cpp

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>

#include "kamping/named_parameter_types.hpp"

namespace kamping::internal {
/// @addtogroup kamping_utility
/// @{

/// @brief Trait struct used to determine the underlying type and value of the parameter type of an object with a
/// parameter type. (This is a building block to enable plugins to have their own named parameters).
/// @tparam Arg Type for which the parameter type unwrapping is done.
template <typename Arg>
struct ParameterTypeUnwrapping {
    using type = std::remove_cv_t<std::remove_reference_t<decltype(std::remove_reference_t<Arg>::parameter_type
    )>>; ///< Type of the underlying parameter type.
    static constexpr type value =
        std::remove_reference_t<Arg>::parameter_type; ///< Value of the underlying parameter type.
};

/// @brief Trait struct used to determine the underlying type and value of the parameter type of an
/// std::integral_constant wrapping a parameter type. (This is a building block to enable plugins to have their own
/// named parameters).
///
/// @tparam T Type of the parameter type.
/// @tparam v Value of the parameter type.
template <typename T, T v>
struct ParameterTypeUnwrapping<std::integral_constant<T, v>> {
    using type                  = T; ///< Type of the underlying parameter type.
    static constexpr type value = v; ///< Value of the underlying parameter type.
};

/// @brief Trait for the underlying type of the parameter type of \tparam Arg.
template <typename Arg>
using parameter_type_t = typename ParameterTypeUnwrapping<Arg>::type;

/// @brief Trait for the underlying value of the parameter type of \tparam Arg.
template <typename Arg>
inline constexpr parameter_type_t<Arg> parameter_type_v = ParameterTypeUnwrapping<Arg>::value;

/// @brief Check whether two objects have the same parameter type.
///
/// @tparam T First type.
/// @tparam U Second type.
/// @return True iff T's and U's underlying parameter type are equal.
template <typename T, typename U>
constexpr bool has_same_parameter_type() {
    if constexpr (std::is_same_v<parameter_type_t<T>, parameter_type_t<U>>) {
        return parameter_type_v<T> == parameter_type_v<U>;
    } else {
        return false;
    }
}

/// @brief Base case if there are no parameters: always returns max index indicating that the parameter was not found.
/// @tparam ParameterTypeConstant Type and value of the parameter type (required for parameter selection within
/// plugins).
/// @tparam Index Index of current argument to evaluate (ignored).
/// @return \c std::numeric_limits<size_t>::max().
template <typename ParameterTypeConstant, size_t Index>
constexpr size_t find_pos() {
    return std::numeric_limits<size_t>::max();
}

/// @brief Returns the Index parameter if the parameter type of Arg matches the requested parameter type. If not, this
/// fails to compile.
///
/// This is the base case of the recursion.
///
/// @tparam ParameterTypeConstant Type and value of the parameter type (required for parameter selection within
/// plugins).
/// @tparam Index Index of current argument to evaluate.
/// @tparam Arg Argument to evaluate.
/// @return The index
/// @return \c std::numeric_limits<size_t>::max() if not found
template <typename ParameterTypeConstant, size_t Index, typename Arg>
constexpr size_t find_pos() {
    constexpr bool found_arg = has_same_parameter_type<Arg, ParameterTypeConstant>();
    // when we do not find the parameter type here, it is not given
    // a we fail to compile with a useful message
    return found_arg ? Index : std::numeric_limits<size_t>::max();
}

/// @brief Returns position of first argument in Args with Trait trait.
///
/// @tparam ParameterTypeConstant Type and value of the parameter type (required for parameter selection within
/// plugins).
/// @tparam Index Index of current argument to evaluate.
/// @tparam Arg Argument to evaluate.
/// @tparam Arg2 The next argument.
/// @tparam Args All remaining arguments.
/// @return Position of first argument with matched trait.
/// @return \c std::numeric_limits<size_t>::max() if not found
template <typename ParameterTypeConstant, size_t Index, typename Arg, typename Arg2, typename... Args>
constexpr size_t find_pos() {
    if constexpr (has_same_parameter_type<Arg, ParameterTypeConstant>()) {
        return Index;
    } else {
        // We need to unpack the next two arguments, so we can unambiguously check for the case
        // of a single remaining argument.
        return find_pos<ParameterTypeConstant, Index + 1, Arg2, Args...>();
    }
}

/// @brief Returns parameter with requested parameter type.
///
/// @tparam ParameterTypeConstant Type and value of the parameter type for which a parameter should be found(required
/// for parameter selection within plugins).
/// @tparam Args All parameter types to be searched for type `ParameterTypeConstant::value`.
/// @param args All parameters from which a parameter with the correct type is selected.
/// @returns The first parameter whose type has the requested parameter type.
template <typename ParameterTypeConstant, typename... Args>
auto& select_parameter_type(Args&... args) {
    constexpr size_t selected_index = find_pos<ParameterTypeConstant, 0, Args...>();
    static_assert(selected_index < sizeof...(args), "Could not find the requested parameter type.");
    return std::get<selected_index>(std::forward_as_tuple(args...));
}

/// @brief "Specialization" for internal::ParameterType.
template <ParameterType parameter_type, typename... Args>
auto& select_parameter_type(Args&... args) {
    return select_parameter_type<std::integral_constant<internal::ParameterType, parameter_type>, Args...>(args...);
}

/// @brief Returns parameter with requested parameter type.
///
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Args All parameter types to be searched for type `parameter_type`.
/// @param tuple std::tuple with containing all parameters from which a parameter with the correct type is selected.
/// @returns The first parameter whose type has the requested parameter type.
template <ParameterType parameter_type, typename... Args>
auto& select_parameter_type_in_tuple(std::tuple<Args...>& tuple) {
    constexpr size_t selected_index = find_pos<std::integral_constant<ParameterType, parameter_type>, 0, Args...>();
    static_assert(selected_index < sizeof...(Args), "Could not find the requested parameter type.");
    return std::get<selected_index>(tuple);
}

/// @brief Returns parameter with requested parameter type.
///
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Args All parameter types to be searched for type `parameter_type`.
/// @param tuple std::tuple with containing all parameters from which a parameter with the correct type is selected.
/// @returns The first parameter whose type has the requested parameter type.
template <ParameterType parameter_type, typename... Args>
auto const& select_parameter_type_in_tuple(std::tuple<Args...> const& tuple) {
    constexpr size_t selected_index = find_pos<std::integral_constant<ParameterType, parameter_type>, 0, Args...>();
    static_assert(selected_index < sizeof...(Args), "Could not find the requested parameter type.");
    return std::get<selected_index>(tuple);
}

/// @brief Type of Buffer with requested \tparam parameter_type
///
/// @tparam TParameterType Type of the parameter type (required for parameter selection within plugins).
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Args All parameter types to be searched for type `parameter_type`.
template <ParameterType parameter_type, typename... Args>
using buffer_type_with_requested_parameter_type = std::
    tuple_element_t<find_pos<std::integral_constant<ParameterType, parameter_type>, 0, Args...>(), std::tuple<Args...>>;

/// @brief Checks if parameter with requested parameter type exists.
///
/// @tparam ParameterTypeConstant Type and value of the parameter type to be searched(required
/// for parameter selection within plugins).
/// @tparam Args All parameter types to be searched.
/// @return \c true iff. `Args` contains a parameter of type `ParameterTypeConstant::value`.
template <typename ParameterTypeConstant, typename... Args>
constexpr bool has_parameter_type() {
    return find_pos<ParameterTypeConstant, 0, Args...>() < sizeof...(Args);
}

/// @brief Checks if parameter with requested parameter type exists.
///
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Args All parameter types to be searched.
/// @return \c true iff. `Args` contains a parameter of type `parameter_type`.
template <ParameterType parameter_type, typename... Args>
constexpr bool has_parameter_type() {
    return has_parameter_type<std::integral_constant<internal::ParameterType, parameter_type>, Args...>();
}

/// @brief Helper struct needed to retrieve the types stored in a std::tuple for the has_parameter_type check.
///
template <typename>
struct has_parameter_helper {};

/// @brief Checks if parameter with requested parameter type exists. Wrapper using the functionality from \ref
/// kamping::internal::has_parameter_type() disassembling a std::tuple passed as parameter.
///
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Args All parameter types to be searched.
/// @return \c true iff. `Args` contains a parameter of type `parameter_type`.
template <ParameterType parameter_type, typename... Args>
constexpr bool has_parameter_type_in_tuple_impl(has_parameter_helper<std::tuple<Args...>> /*args*/) {
    return has_parameter_type<parameter_type, Args...>();
}

/// @brief Checks if parameter with requested parameter type exists.
///
/// @tparam parameter_type The parameter type with which a parameter should be found.
/// @tparam Tuple Intended: std::tuple<Args...> containing all types to be searched.
/// @return \c true iff. `Args` contains a parameter of type `parameter_type`.
template <ParameterType parameter_type, typename Tuple>
constexpr bool has_parameter_type_in_tuple() {
    return has_parameter_type_in_tuple_impl<parameter_type>(has_parameter_helper<Tuple>{});
}

/// @brief Checks if parameter with requested parameter type exists, if not constructs a default value.
///
///
/// @tparam ParameterTypeConstant Type and value of the parameter type for which a parameter should be found(required
/// for parameter selection within plugins).
/// @tparam Args All parameter types to be searched for parameter type `ParameterTypeConstant::value`.
/// @tparam DefaultParameterType The type of the default parameter to be constructed.
/// @tparam DefaultArguments The types of parameters passed to the constructor \c DefaultParameterType.
/// @param default_arguments Tuple of the arguments passed to the constructor of \c DefaultParameterType.
/// @param args All parameters from which a parameter with the correct type is selected.
/// @return The first parameter whose type has the requested parameter type or the constructed default parameter if
/// none is found.
template <typename ParameterTypeConstant, typename DefaultParameterType, typename... DefaultArguments, typename... Args>
decltype(auto) select_parameter_type_or_default(std::tuple<DefaultArguments...> default_arguments, Args&... args) {
    static_assert(
        std::is_constructible_v<DefaultParameterType, DefaultArguments...>,
        "The default parameter cannot be constructed from the provided arguments"
    );
    if constexpr (has_parameter_type<ParameterTypeConstant, Args...>()) {
        constexpr size_t selected_index = find_pos<ParameterTypeConstant, 0, Args...>();
        return std::get<selected_index>(std::forward_as_tuple(args...));
    } else {
        static_assert(!std::is_reference_v<DefaultParameterType>, "DefaultParameterType must not be a reference.");
        return std::make_from_tuple<DefaultParameterType>(std::move(default_arguments));
    }
}

/// @brief "Specialization" for internal::ParameterType.
template <ParameterType parameter_type, typename DefaultParameterType, typename... DefaultArguments, typename... Args>
decltype(auto) select_parameter_type_or_default(std::tuple<DefaultArguments...> default_arguments, Args&... args) {
    return select_parameter_type_or_default<
        std::integral_constant<internal::ParameterType, parameter_type>,
        DefaultParameterType>(std::move(default_arguments), args...);
}

/// @}
} // namespace kamping::internal
