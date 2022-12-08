// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

/// @file
/// @brief Macros for generating concept-like type traits to check for member functions of objects.

#pragma once

#include <type_traits>

/// @brief Macro for generating has_member_xxx and has_member_xxx_v templates.
/// They return true if the type given as template parameter has a member function or class member with the provided
/// name.
///
/// Example:
/// \code
/// KAMPING_MAKE_HAS_MEMBER(baz);
/// struct Foo {
///   int baz(char);
/// };
/// static_assert(has_member_baz_v<Foo>)
/// \endcode
///
/// Explanation:
/// - to obtain \c value, the static member function \c test is instantiated using the given type \c Type.
/// - therefore, we try to obtain the the type of the address of the member of that class
/// - if that member does not exist, we can not obtain the decltype, and cannot instantiate \c std::void_t<>, which
/// fails to initialize the whole function \c test(int)
/// - then, the next best instantiation is \c test(long), which returns \c std::false_type.
/// - if we find the requested member, we get \c std::true_type
/// - \c test has \c int and \c long overloads to resolve ambiguity. Passing 0 to \c test ensure that we first try to
/// instantiate the \c true variant.
#define KAMPING_MAKE_HAS_MEMBER(Member)                                                                         \
    template <typename Type, typename... MemberArgs>                                                            \
    class has_member_##Member {                                                                                 \
        template <                                                                                              \
            typename C,                                                                                         \
            typename = std::void_t<decltype(std::declval<C>().Member(std::declval<MemberArgs>()...))>>          \
        static auto test(int) -> std::true_type;                                                                \
        template <typename C>                                                                                   \
        static auto test(long) -> std::false_type;                                                              \
        template <                                                                                              \
            typename C,                                                                                         \
            typename... TemplateParams,                                                                         \
            typename = std::void_t<                                                                             \
                decltype(std::declval<C>().template Member<TemplateParams...>(std::declval<MemberArgs>()...))>> \
        static auto test_with_template_params(int) -> std::true_type;                                           \
        template <typename C, typename... TemplateParams>                                                       \
        static auto test_with_template_params(long) -> std::false_type;                                         \
                                                                                                                \
    public:                                                                                                     \
        static constexpr bool value = decltype(test<Type>(0))::value;                                           \
                                                                                                                \
        template <typename... TemplateParams>                                                                   \
        static constexpr bool value_with_template_params =                                                      \
            decltype(test_with_template_params<Type, TemplateParams...>(0))::value;                             \
    };                                                                                                          \
                                                                                                                \
    template <typename Type, typename... MemberArgs>                                                            \
    [[maybe_unused]] static constexpr bool has_member_##Member##_v = has_member_##Member<Type, MemberArgs...>::value;
