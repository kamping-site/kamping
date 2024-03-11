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
#include "kamping/named_parameter_types.hpp"
#include "kamping/serialization.hpp"

// The following macros look strange since they use a GNU extension that becomes obsolete with C++-20.
// The extension is supported by all major compilers.
// Semantic is as follows: if the variadic parameters of a macro is empty,
//
// , ##__VA_ARGS__
//
// expands to nothing, i.e., the comma vanishes. Otherwise, the comma stays.
// This is required because we overload macros based on their number of arguments; however, the preprocessor
// considers an empty argument to be one argument, i.e., if we have a variadic macro
//
// #define M(...)
//
// and call it as M(), we actually call it with one argument: the empty argument. But since we decide what to do with M
// based on the number of arguments passed to it, we need a way to distinguish between "1 empty argument" and "1
// actual argument".
//
// Using this trick, "zero arguments" as in "1 argument, but empty" resolves to <empty>, while 1 actual argument
// resolves to ", <argument>", i.e., 2 arguments.
//
// However, this leads to the situation where the first argument of this macro is always empty; that's why we have
// a lot of "ignore" parameters in the remaining macros of this file.

/// @brief Wrapper to pass (possibly empty) list of parameter type names as required parameters to \c
/// KAMPING_CHECK_PARAMETERS.
/// Note that this macro only takes the *name* of parameter types, i.e., instead of using
/// `kamping::internal::ParameterType::send_buf`, only pass `send_buf` to this macro.
#define KAMPING_REQUIRED_PARAMETERS(...) , ##__VA_ARGS__

/// @brief Wrapper to pass (possibly empty) list of parameter type names as optional parameters to \c
/// KAMPING_CHECK_PARAMETERS.
/// Note that this macro only takes the *name* of parameter types, i.e., instead of using
/// `kamping::internal::ParameterType::send_buf`, only pass `send_buf` to this macro.
#define KAMPING_OPTIONAL_PARAMETERS(...) , ##__VA_ARGS__

/// @brief Assertion macro that checks if passed parameters are correct, i.e., all parameter types are unique, all
/// required parameters are provided, and no unused parameter is passed. Also checks that all parameter types are
/// r-value references.
///
/// The macro *only* expects the parameter type, i.e., a member name of the `kamping::internal::ParameterType` enum
/// *without the name of the enum*. For instance,
/// ```c++
/// KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, recv_buf), KAMPING_OPTIONAL_PARAMETERS())
/// ```
/// checks that the parameter pack `Args` contains members of type `kamping::internal::ParameterType::send_buf` and type
/// `kamping::internal::ParameterType::recv_buf`.
///
/// Note that expanding the macro into a `do { ... } while(false)` pseudo-loop is a common trick to make a macro
/// "act like a statement". Otherwise, it would have surprising effects if the macro is used inside a `if` branch
/// without braces.
///
/// @param args A parameter pack with all parameter types passed to the function. Note that this is only the name of the
/// parameter pack *without trailing `...`*.
/// @param required A list of required parameter type names wrapped in a KAMPING_REQUIRED_PARAMETERS macro.
/// @param optional A list of optional parameter type names wrapped in a KAMPING_OPTIONAL_PARAMETERS macro.
#define KAMPING_CHECK_PARAMETERS(args, required, optional)                                                          \
    do {                                                                                                            \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, required);                                     \
                                                                                                                    \
        using required_parameters_types = typename kamping::internal::parameter_types_to_integral_constants<        \
            KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(required)>::type;                                         \
        using optional_parameters_types = typename kamping::internal::parameter_types_to_integral_constants<        \
            KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(optional)>::type;                                         \
        using parameter_types = typename kamping::internal::parameters_to_integral_constant<args...>::type;         \
        static_assert(                                                                                              \
            kamping::internal::                                                                                     \
                has_no_unused_parameters<required_parameters_types, optional_parameters_types, args...>::assertion, \
            "There are unsupported parameters, only support required "                                              \
            "parameters " KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(required                                       \
            ) " and optional parameters " KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(optional)                      \
        );                                                                                                          \
        static_assert(kamping::internal::all_unique_v<parameter_types>, "There are duplicate parameter types.");    \
    } while (false)

/// @cond IMPLEMENTATION

// Used to stringify variadic parameters:
// KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(a, b, c) returns the string "a, b, c"
#define KAMPING_PARAMETER_CHECK_HPP_EVAL_STRINGIFY(ignore, ...) "[" #__VA_ARGS__ "]"

// In the following, we implement variadic macros that do something for each of their arguments:
// - KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(...) prepends each argument by "kamping::internal::ParameterType::"
// - KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, ...) generates a static assert for each of its
//   arguments to ensure that args... contains a parameter of that type.
//
// Since doing something "for each" argument of a variadic macro is unsupported by the preprocessor, we use two
// hacks to implement these macros:
// - Instead of a "for each" loop, we implement macros Xi, for 1 <= i <= 9, such that Xi takes i arguments
//   and generates the same code as a "for each" loop would generate for these i arguments.
// - A dispatch macro chooses the right Xi macro depending on the number of arguments passed to the dispatch macro.
//
// First, we define the macros for various number of arguments:
//
// ```
// #define X0 [...]
// #define X1(a) [...]
// #define X2(a, b) [...]
// ```
//
// Now, we need a "dispatch" macro `X` that can take 0, 1 or 2 arguments and resolve to `X0`, `X1` or `X2`. While we
// can't make the macro to take between 0 and 2 arguments, we can define it as a variadic macro:
//
// ```
// #define X(...) [... magic that expands to X2, X1 or X0 depending on the number of arguments passed to X ...]
// ```
//
// To implement this macro, we first need a helper:
//
// ```
// #define DISPATCH(x2, x1, x, ...) x
// ```
//
// `DISPATCH` takes at least 3 arguments and substitutes to whatever we pass as 3rd argument, e.g.:
//
// ```
// DISPATCH(a, b, c, d, e, f) // becomes c
// DISPATCH(0, 1, X2(0, 1), X1(0, 1), X0) // becomes X2(0, 1)
// DISPATCH(0, X2(0), X1(0), X0) // becomes X1(0)
// DISPATCH(X2(), X1(), X0) // becomes X0
// ```
//
// At least in theory -- if one strictly adheres to the C++ standard, `DISPATCH` actually takes at least 4 arguments,
// since the `...` parameter may not be empty. Thus, in our implementation, we pass another dummy argument to the
// `DISPATCH` invocation, e.g:
// ```
// DISPATCH(X2(), X1(), X0, ignore)
// ```
//
// We can use that to implement `X`:
//
// ```
// #define X(...) DISPATCH(__VA_ARGS__, X2(__VA_ARGS__), X1(__VA_ARGS__), X0, ignore)
// ```
//
// `__VA_ARGS__` expands to whatever arguments we pass to `X`. Thus, if we pass 2 arguments to `X`, it also expands to
// two arguments. If we pass 1 argument to `X`, it expands to 1 argument. Thus, we can "move" the correct implementation
// for `X` to be the 3rd argument passed to `DISPATCH`:
//
// * `X(0, 1)` becomes `DISPATCH(0, 1, X2(0, 1), X1(0, 1), X0, ignore)` becomes `X2(0, 1)`
//
// * `X(0)` becomes `DISPATCH(0, X2(0), X1(0), X0, ignore)` becomes `X1(0)`
//
// Since KAMPING_REQUIRED_PARAMETERS and KAMPING_OPTIONAL_PARAMETERS always resolve to at least one argument (see
// description above), this is sufficient.

// DISPATCH helper macro as described above
#define KAMPING_PARAMETER_CHECK_HPP_SELECT10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y, ...) y

// Adds the prefix "kamping::internal::ParameterType::" to each of its arguments (up to 10 arguments)
// I.e., turns "send_buf, recv_buf" into "kamping::internal::ParameterType::send_buf,
// kamping::internal::ParameterType::recv_buf"
//
// We do this because we need both versions of the parameter types: to print nice error messages, we only need the names
// of the types without preceding `kamping::internal::ParameterType::`; to implement the checks, we need the actual
// name of the symbol to generate valid C++ code.
//
// Note that argument "ignore" argument in this macro is required because the "..." parameter of
// KAMPING_PARAMETER_CHECK_HPP_SELECT10 may not be empty.
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX_PARAMETERS(...) \
    KAMPING_PARAMETER_CHECK_HPP_SELECT10(                  \
        __VA_ARGS__,                                       \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX9(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX8(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX7(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX6(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX5(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX4(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX3(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX2(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX1(__VA_ARGS__),  \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX0(__VA_ARGS__),  \
        ignore                                             \
    )

#define KAMPING_PARAMETER_CHECK_HPP_PREFIX0(ignore)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x1) kamping::internal::ParameterType::x1
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX2(ignore, x1, x2) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x1), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x2)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX3(ignore, x1, x2, x3) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX2(ignore, x1, x2), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x3)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX4(ignore, x1, x2, x3, x4) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX3(ignore, x1, x2, x3), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x4)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX5(ignore, x1, x2, x3, x4, x5) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX4(ignore, x1, x2, x3, x4), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x5)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX6(ignore, x1, x2, x3, x4, x5, x6) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX5(ignore, x1, x2, x3, x4, x5), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x6)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX7(ignore, x1, x2, x3, x4, x5, x6, x7) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX6(ignore, x1, x2, x3, x4, x5, x6), KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x7)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX8(ignore, x1, x2, x3, x4, x5, x6, x7, x8) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX7(ignore, x1, x2, x3, x4, x5, x6, x7),        \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x8)
#define KAMPING_PARAMETER_CHECK_HPP_PREFIX9(ignore, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    KAMPING_PARAMETER_CHECK_HPP_PREFIX8(ignore, x1, x2, x3, x4, x5, x6, x7, x8),        \
        KAMPING_PARAMETER_CHECK_HPP_PREFIX1(ignore, x9)

// Generate code that checks that each of the given parameter types are present in args...
// Usage: KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(Args, send_buf, recv_buf)
// Checks that Args... has parameters for kamping::internal::ParameterType::send_buf and
// kamping::internal::ParameterType::recv_buf
//
// Note that the "ignore" argument in this macro is required because the "..." parameter of
// KAMPING_PARAMETER_CHECK_HPP_SELECT10 may not be empty.
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETERS(args, ...)          \
    KAMPING_PARAMETER_CHECK_HPP_SELECT10(                                          \
        __VA_ARGS__,                                                               \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER9(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, __VA_ARGS__), \
        KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER0(args, __VA_ARGS__), \
        ignore                                                                     \
    )

#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER0(args, ignore)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x1)                \
    static_assert(                                                                              \
        kamping::internal::has_parameter_type<kamping::internal::ParameterType::x1, args...>(), \
        "Missing required parameter " #x1                                                       \
    );
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, ignore, x1, x2) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x1);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x2)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, ignore, x1, x2, x3) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER2(args, ignore, x1, x2);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x3)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, ignore, x1, x2, x3, x4) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER3(args, ignore, x1, x2, x3);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x4)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, ignore, x1, x2, x3, x4, x5) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER4(args, ignore, x1, x2, x3, x4);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x5)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, ignore, x1, x2, x3, x4, x5, x6) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER5(args, ignore, x1, x2, x3, x4, x5);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x6)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, ignore, x1, x2, x3, x4, x5, x6, x7) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER6(args, ignore, x1, x2, x3, x4, x5, x6);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x7)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, ignore, x1, x2, x3, x4, x5, x6, x7, x8) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER7(args, ignore, x1, x2, x3, x4, x5, x6, x7);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x8)
#define KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER9(args, ignore, x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER8(args, ignore, x1, x2, x3, x4, x5, x6, x7, x8);        \
    KAMPING_PARAMETER_CHECK_HPP_ASSERT_REQUIRED_PARAMETER1(args, ignore, x9)

/// @brief Assertion macro that checks if a parameter type is not present in the arguments.
/// Outputs a static assert if the parameter type is present in the arguments with text "Parameter type <parameter_type>
/// is not supported <whatfor>."
#define KAMPING_UNSUPPORTED_PARAMETER(args, parameter_type, whatfor)                                         \
    static_assert(                                                                                           \
        !kamping::internal::has_parameter_type<kamping::internal::ParameterType::parameter_type, args...>(), \
        "Parameter type " #parameter_type " is not supported " #whatfor "."                                  \
    )

/// @endcond

namespace kamping::internal {
/// @brief Struct wrapping a check that verifies that no unused parameters are part of the arguments.
///
/// @tparam RequiredParametersTuple All required kamping::internal::ParameterType passed as \c
/// std::integral_constant in an \c std::tuple.
/// @tparam OptionalParametersTuple All optional kamping::internal::ParameterType passed as \c
/// std::integral_constant in an \c std::tuple.
/// @tparam Args Arguments passed to the function that calls this check, i.e., the different parameters.
template <typename RequiredParametersTuple, typename OptionalParametersTuple, typename... Args>
struct has_no_unused_parameters {
    /// @brief Concatenation of required and optional parameters.
    using all_available_parameters = decltype(std::tuple_cat(RequiredParametersTuple{}, OptionalParametersTuple{}));

    /// @brief Get total number of different parameters (passed, required, and optional).
    ///
    /// This check works similar to has_all_required_parameters. Here, we "iterate" over all parameters,
    /// i.e., \c RequiredParametersTuple and \c OptionalParametersTuple and check which parameters are
    /// not(!) passed as \c Args. Then, we add this number to the size of Args. If this number is
    /// greater than the total number of (required and optional) parameters, there are unused
    /// parameters.
    ///
    /// @tparam Indices Index sequence used to unpack all required parameters in \c ParametersTuple.
    /// @param indices The parameter is only required to deduce the template parameter.
    /// @return The number of different parameters (passed, optional, and required).
    template <size_t... Indices>
    static constexpr auto number_distinct_parameters(std::index_sequence<Indices...> indices [[maybe_unused]]) {
        return std::tuple_size_v<decltype(std::tuple_cat(
                   std::conditional_t<
                       !has_parameter_type<std::tuple_element_t<Indices, all_available_parameters>::value, Args...>(),
                       std::tuple<std::tuple_element_t<Indices, all_available_parameters>>,
                       std::tuple<>>{}...
               ))> + sizeof...(Args);
    }

    /// @brief \c true if and only if no unused parameter can be found in \c Args.
    static constexpr bool assertion =
        (std::tuple_size_v<all_available_parameters> >= number_distinct_parameters(
             std::make_index_sequence<std::tuple_size_v<all_available_parameters>>{}
         ));

}; // struct has_no_unused_parameters

/// @brief Base wrapper (\c std::integral_constant) to test if all types of a tuple are unique.
/// @tparam Tuple Tuple for which is it checked whether all types are unique.
template <typename Tuple>
struct all_unique : std::true_type {};

/// @brief Recursive wrapper (\c std::integral_constant) to test if all types of a tuple are unique.
/// This is done by checking for each type whether the type occurs in the types of the tuple to the
/// right. If this is true for any type/position, the types in the tuple are not unique.
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

/// @brief Wrapper to get a tuple of \c std::integral_constant for each kamping::internal::ParameterType
/// passed as template parameter that are extracted as tuple of \c std::integral_constant.
/// @tparam Parameters Passed kamping::internal::ParameterType.
template <ParameterType... ParameterTypes>
struct parameter_types_to_integral_constants {
    /// @brief Type of the tuple.
    using type =
        decltype(std::tuple_cat(std::tuple<typename parameter_type_to_integral_constant<ParameterTypes>::type>{}...));
};

/// @brief Wrapper to get a tuple of \c std::integral_constant for each kamping::internal::ParameterType
/// of the parameters.
/// @tparam Parameters Passed parameters for which the kamping::internal::ParameterType are extracted as
/// \c std::integral_constant in a tuple.
template <typename... Parameters>
struct parameters_to_integral_constant {
    /// @brief Type of the tuple.
    using type = decltype(std::tuple_cat(
        std::tuple<typename parameter_type_to_integral_constant<Parameters::parameter_type>::type>{}...
    ));
};

/// @brief Checks if a data buffer with requested parameter type exists and it is an input parameter (i.e. its content
/// does not have to be computed/deduced by KaMPIng).
///
/// @tparam parameter_type The parameter type for which a parameter should be found.
/// @tparam Args All parameter types to be searched.
/// @return \c true iff. `Args` contains a parameter of type `parameter_type` and this parameter is not an output
/// buffer.
template <ParameterType parameter_type, typename... Args>
static constexpr bool is_parameter_given_as_in_buffer = []() {
    constexpr size_t found_pos = find_pos<std::integral_constant<ParameterType, parameter_type>, 0, Args...>();
    if constexpr (found_pos >= sizeof...(Args)) {
        return false;
    } else {
        using FoundType = std::tuple_element_t<found_pos, std::tuple<Args...>>;
        return !FoundType::is_out_buffer;
    }
}();

/// @brief Checks if the buffer has to be computed by kamping, i.e. if it is an output parameter or the buffer has been
/// allocated by KaMPIng.
/// @tparam BufferType The buffer type to be checked
template <typename BufferType>
static constexpr bool has_to_be_computed =
    std::remove_reference_t<BufferType>::is_out_buffer || std::remove_reference_t<BufferType>::is_lib_allocated;

/// @brief Checks if all buffers have to be computed by kamping, i.e., if all buffers are output parameters of the
/// buffers have been allocated by kamping.
/// @tparam BufferTypes Any number of buffers types to be checked.
template <typename... BufferTypes>
static constexpr bool all_have_to_be_computed = (has_to_be_computed<BufferTypes> && ...);

/// @brief Checks if any of the buffers have to be computed by kamping, i.e., if at least one buffer is an output
/// parameter or has been allocated by kamping.
/// @tparam BufferTypes Any number of buffers to be checked.
template <typename... BufferTypes>
static constexpr bool any_has_to_be_computed = (has_to_be_computed<BufferTypes> || ...);

/// @brief Checks if \p DataBufferType is a serialization buffer.
template <typename DataBufferType>
static constexpr bool buffer_uses_serialization = internal::is_serialization_buffer_v<
    typename std::remove_const_t<std::remove_reference_t<DataBufferType>>::MemberTypeWithConstAndRef>;

} // namespace kamping::internal
