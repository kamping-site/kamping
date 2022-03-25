// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
/// @brief Template magic to check named parameters passed to wrappers at compile time.

#pragma once

#include <tuple>
#include <type_traits>

#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_type_definitions.hpp"

/// @brief Wrapper to pass (possibly empty) list of parameters as required parameters to \c KAMPING_CHECK_PARAMETERS.
#define KAMPING_REQUIRED_PARAMETERS(...) __VA_ARGS__

/// @brief Wrapper to pass (possibly empty) list of parameters as optional parameters to \c KAMPING_CHECK_PARAMETERS.
#define KAMPING_OPTIONAL_PARAMETERS(...) __VA_ARGS__

#define KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(...) #__VA_ARGS__

/// @brief Assertion macro that checks if passed parameters are correct, i.e., all parameter types are unique, all
/// required parameters are provided, and on unused parameter is passed.
///
/// The \c REQUIRED parameter should be passed as \c KAMPING_REQUIRED_PARAMETERS and the \c OPTIONAL parameter should be
/// passed as KAMPING_OPTIONAL_PARAMETERS.
#define KAMPING_CHECK_PARAMETERS(args, required, optional)                                                      \
    do {                                                                                                        \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, required);                                 \
                                                                                                                \
        using required_parameters_types =                                                                       \
            typename parameter_types_to_integral_constants<KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(       \
                required)>::type;                                                                               \
        using optional_parameters_types =                                                                       \
            typename parameter_types_to_integral_constants<KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(       \
                optional)>::type;                                                                               \
        using parameter_types = typename parameters_to_integral_constant<args...>::type;                        \
        static_assert(                                                                                          \
            has_no_unused_parameters<required_parameters_types, optional_parameters_types, args...>::assertion, \
            "There are unsupported parameters, only support required "                                          \
            "parameters " KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(                                           \
                required) " and optional parameters " KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(optional));    \
        static_assert(all_unique_v<parameter_types>, "There are duplicate parameter types.");                   \
    } while (false)

/// @cond IMPLEMENTATION

// In the following, we implement variadic macros that do something for each of their arguments:
// - KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(...) prepends each argument by "internal::kamping::ParameterType::"
// - KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, ...) generates a static assert for each of its
//   arguments to ensure that args... contains a parameter of that type.
//
// Since doing something "for each" argument of a variadic macro is unsupported by the preprocessor, we use two
// hacks to implement these macros:
// - Instead of a "for each" loop, we implement macros Xi, for 1 <= i <= 9, such that Xi takes i arguments
//   and generates the same code as a "for each" loop would generate for these i arguments.
// - A dispatch macro chooses the right Xi macro depending on the number of arguments passed to the dispatch macro.
//
// Note that Xi is short for
// - KAMPING_PARAMETER_CHECK_HPP_PREFIXi
// or
// - KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERi
//
// For instance, KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(a, b, c) would dispatch to
// KAMPING_PARAMETER_CHECK_HPP_PREFIX3(a, b, c), which in turn prefixes its 3 arguments as described above.
//
// This works through the helper macro KAMPING_PARAMETER_CHECK_HPP_SELECT10, which takes at least 11 arguments
// and always "returns" its 11-th argument.
// Thus, do implement the dispatch macro, we use the following hack:
// - Take variadic arguments; those are available in __VA_ARGS__
// - Call KAMPING_PARAMETER_CHECK_HPP_SELECT10 with the following arguments:
//   * __VA_ARGS__ = all variadic parameters
//   * X9(__VA_ARGS__)
//   * X8(__VA_ARGS__)
//   * ...
//   * X1(__VA_ARGS__)
// - Thus, the "11-th argument of KAMPING_PARAMETER_CHECK_HPP_SELECT10" depends on the number of arguments
//   __VA_ARGS__ expands to -- and thus, on the number of arguments passed to the dispatch macro.
// - In other words, we "push" the right implementation to the right parameter position of
//   KAMPING_PARAMETER_CHECK_HPP_SELECT10.
//
// That's all there is do it -- in theory. In practice, we need another hack:
// - We always pass "ignore" as a last argument to KAMPING_PARAMETER_CHECK_HPP_SELECT10. Otherwise, if the
//   dispatch macro was called with just one argument, we would call KAMPING_PARAMETER_CHECK_HPP_SELECT10
//   with exactly 11 arguments (1 empty argument + 1 variadic argument + X1 + ... + X9), thus leaving the
//   "..." of KAMPING_PARAMETER_CHECK_SELECT10 empty. But this is not allowed, even if the macro ignores
//   its variadic arguments.

#define KAMPING_PARAMETER_CHECK_HPP_SELECT10(x1, x2, x3, x4, x5, x6, x7, x8, x9, y, ...) y

#define KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(...)                                                  \
    KAMPING_PARAMETER_CHECK_HPP_SELECT10(                                                                   \
        __VA_ARGS__, KAMPING_PARAMETER_CHECK_HPP_PREFIX9(__VA_ARGS__),                                      \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX8(__VA_ARGS__), KAMPING_PARAMETER_CHECK_HPP_PREFIX7(__VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX6(__VA_ARGS__), KAMPING_PARAMETER_CHECK_HPP_PREFIX5(__VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX4(__VA_ARGS__), KAMPING_PARAMETER_CHECK_HPP_PREFIX3(__VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX2(__VA_ARGS__), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(__VA_ARGS__), ignore)

#define KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x1) kamping::internal::ParameterType::x1
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX2(x1, x2) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x1), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x2)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX3(x1, x2, x3) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX2(x1, x2), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x3)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX4(x1, x2, x3, x4) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX3(x1, x2, x3), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x4)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX5(x1, x2, x3, x4, x5) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX4(x1, x2, x3, x4), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x5)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX6(x1, x2, x3, x4, x5, x6) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX5(x1, x2, x3, x4, x5), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x6)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX7(x1, x2, x3, x4, x5, x6, x7) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX6(x1, x2, x3, x4, x5, x6), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x7)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX8(x1, x2, x3, x4, x5, x6, x7, x8) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX7(x1, x2, x3, x4, x5, x6, x7), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x8)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX9(x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX8(x1, x2, x3, x4, x5, x6, x7, x8), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(x9)

#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, ...)                       \
    KAMPING_PARAMETER_CHECK_HPP_SELECT10(                                                       \
        __VA_ARGS__, KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER9(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, __VA_ARGS__),              \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, __VA_ARGS__), ignore)

#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x1)                                          \
    static_assert(                                                                                                \
        kamping::internal::has_all_required_parameters<                                                           \
            kamping::internal::parameter_types_to_integral_constants<kamping::internal::ParameterType::x1>::type, \
            args...>::assertion,                                                                                  \
        "Missing required parameter " #x1);
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, x1, x2) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x1);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x2)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, x1, x2, x3) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, x1, x2);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x3)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, x1, x2, x3, x4) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, x1, x2, x3);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x4)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, x1, x2, x3, x4, x5) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, x1, x2, x3, x4);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x5)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, x1, x2, x3, x4, x5, x6) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, x1, x2, x3, x4, x5);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x6)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, x1, x2, x3, x4, x5, x6, x7) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, x1, x2, x3, x4, x5, x6);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x7)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, x1, x2, x3, x4, x5, x6, x7, x8) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, x1, x2, x3, x4, x5, x6, x7);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x8)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER9(args, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, x1, x2, x3, x4, x5, x6, x7, x8);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, x9)

/// @endcond

namespace kamping::internal {
/// @brief Struct wrapping a check that verifies that all required parameters are part of the arguments.
///
/// @tparam ParametersTuple All required kamping::internal::ParameterType passed as \c std::integral_constant in an \c
/// std::tuple.
/// @tparam Args Arguments passed to the function that calls this check, i.e., the different parameters.
template <typename ParametersTuple, typename... Args>
struct has_all_required_parameters {
    /// @brief Get number of required parameters passed as argument in \c Args.
    ///
    /// To compute the number, we "iterate" over all required template parameters and check if the parameter can be
    /// found (using has_parameter_type() on the arguments \c Args). If this is the case, we add a tuple with the given
    /// parameter type to the result, otherwise, we add a tuple without a type. In the end, we have added a parameter
    /// type for each parameter type that we have found in \c Args. Hence, the size of the resulting tuple is the number
    /// of found parameters.
    ///
    /// @tparam Indices Index sequence used to unpack all required parameters in \c ParametersTuple.
    /// @param N.N. The parameter is only required to deduce the template parameter.
    /// @return The number of required parameters found in \c Args.
    template <size_t... Indices>
    static constexpr auto number_of_required(std::index_sequence<Indices...>) {
        return std::tuple_size_v<decltype(
            std::tuple_cat(std::conditional_t<
                           has_parameter_type<std::tuple_element_t<Indices, ParametersTuple>::value, Args...>(),
                           std::tuple<std::tuple_element_t<Indices, ParametersTuple>>, std::tuple<>>{}...))>;
    }

    /// @brief \c true if and only if all required parameters can be found in \c Args.
    static constexpr bool assertion =
        (std::tuple_size_v<
             ParametersTuple> == number_of_required(std::make_index_sequence<std::tuple_size_v<ParametersTuple>>{}));
}; // struct has_all_required_parameters

/// @brief Struct wrapping a check that verifies that no unused parameters are part of the arguments.
///
/// @tparam RequiredParametersTuple All required kamping::internal::ParameterType passed as \c std::integral_constant in
/// an \c std::tuple.
/// @tparam OptionalParametersTuple All optional kamping::internal::ParameterType passed as \c std::integral_constant in
/// an \c std::tuple.
/// @tparam Args Arguments passed to the function that calls this check, i.e., the different parameters.
template <typename RequiredParametersTuple, typename OptionalParametersTuple, typename... Args>
struct has_no_unused_parameters {
    using all_available_parameters = decltype(std::tuple_cat(RequiredParametersTuple{}, OptionalParametersTuple{}));

    /// @brief Get total number of different parameters (passed, required, and optional).
    ///
    /// This check works similar to has_all_required_parameters. Here, we "iterate" over all parameters, i.e., \c
    /// RequiredParametersTuple and \c OptionalParametersTuple and check which parameters are not(!) passed as \c Args.
    /// Then, we add this number to the size of Args. If this number is greater than the total number of (required and
    /// optional) parameters, there are unused parameters.
    ///
    /// @tparam Indices Index sequence used to unpack all required parameters in \c ParametersTuple.
    /// @param N.N. The parameter is only required to deduce the template parameter.
    /// @return The number of different parameters (passed, optional, and required).
    template <size_t... Indices>
    static constexpr auto total_number_of_parameter(std::index_sequence<Indices...>) {
        return std::tuple_size_v<decltype(std::tuple_cat(
                   std::conditional_t<
                       !has_parameter_type<std::tuple_element_t<Indices, all_available_parameters>::value, Args...>(),
                       std::tuple<std::tuple_element_t<Indices, all_available_parameters>>,
                       std::tuple<>>{}...))> + sizeof...(Args);
    }

    /// @brief \c true if and only if no unused parameter can be found in \c Args.
    static constexpr bool assertion =
        (std::tuple_size_v<all_available_parameters> >= total_number_of_parameter(
             std::make_index_sequence<std::tuple_size_v<all_available_parameters>>{}));

}; // struct has_no_unused_parameters

/// @brief Base wrapper (\c std::integral_constant) to test if all types of a tuple are unique.
/// @tparam Tuple Tuple for which is it checked whether all types are unique.
template <typename Tuple>
struct all_unique : std::true_type {};

/// @brief Recursive wrapper (\c std::integral_constant) to test if all types of a tuple are unique.
/// This is done by checking for each type whether the type occurs in the types of the tuple to the right. If this is
/// true for any type/position, the types in the tuple are not unique.
///
/// @tparam T Parameter for which we check whether it is contained in the remaining tuple.
/// @tparam Ts Remaining types of the tuple.
template <typename T, typename... Ts>
struct all_unique<std::tuple<T, Ts...>>
    : std::conjunction<std::negation<std::disjunction<std::is_same<T, Ts>...>>, all_unique<std::tuple<Ts...>>> {};

/// @brief \c true if and only if all types of the tuple are unique.
template <typename Tuple>
static constexpr bool all_unique_v = all_unique<Tuple>::value;

/// @brief Wrapper to get an \c std::integral_constant for a kamping::internal::ParameterType.
/// @tparam T kamping::internal::ParameterType that is converted to an \c std::integral_constant.
template <ParameterType T>
struct parameter_type_to_integral_constant {
    /// @brief kamping::internal::ParameterType as \c std::integral_constant.
    using type = std::integral_constant<ParameterType, T>;
};

/// @brief Wrapper to get a tuple of \c std::integral_constant for each kamping::internal::ParameterType passed as
/// template parameter that are extracted as tuple of \c std::integral_constant.
/// @tparam Parameters Passed kamping::internal::ParameterType.
template <ParameterType... ParameterTypes>
struct parameter_types_to_integral_constants {
    /// @brief Type of the tuple.
    using type =
        decltype(std::tuple_cat(std::tuple<typename parameter_type_to_integral_constant<ParameterTypes>::type>{}...));
};

/// @brief Wrapper to get a tuple of \c std::integral_constant for each kamping::internal::ParameterType of the
/// parameters.
/// @tparam Parameters Passed parameters for which the kamping::internal::ParameterType are extracted as \c
/// std::integral_constant in a tuple.
template <typename... Parameters>
struct parameters_to_integral_constant {
    /// @brief Type of the tuple.
    using type = decltype(std::tuple_cat(
        std::tuple<typename parameter_type_to_integral_constant<Parameters::parameter_type>::type>{}...));
};

} // namespace kamping::internal
