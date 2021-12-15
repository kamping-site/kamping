// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>
// TODO Probably rename and reorganize all of this

enum class ptraits { in, out, root, recvCounts, recvDispls };

// trait selector *************************************************************
// returns the index of the first argument type that has the appropriate par_type
template <ptraits trait, size_t I, class Arg, class... Args>
constexpr size_t find_pos() {
    if constexpr (Arg::par_type == trait)
        return I;
    else
        return find_pos<trait, I + 1, Args...>();
}

// returns the first parameter whose type has the appropriate par_type
template <ptraits trait, class... Args>
decltype(auto) select_trait(Args&&... args) {
    return std::move(std::get<find_pos<trait, 0, Args...>()>(std::forward_as_tuple(args...)));
}

template <class T>
struct in_named_tuple {
    T*     ptr;
    size_t size;
};

template <class T>
class in_type_ptr {
public:
    // each class contains its type as par_type (must be known at compile time)
    static constexpr ptraits par_type = ptraits::in;
    using value_type                  = T;

    in_type_ptr(const T* ptr, size_t size) : _ptr(ptr), _size(size) {}

    in_named_tuple<T> get() {
        return {_ptr, _size};
    }

private:
    T*     _ptr;
    size_t _size;
};

template <class T>
class in_type_vec {
public:
    // each class contains its type as par_type (must be known at compile time)
    static constexpr ptraits par_type = ptraits::in;
    using value_type                  = T;

    in_type_vec(std::vector<T>& vec) : _vec(vec) {}

    in_named_tuple<T> get() {
        return {_vec.data(), _vec.size()};
    }

private:
    std::vector<T>& _vec;
};
=======
/// @file
/// @brief Template magic to implement named parameters in cpp
>>>>>>> 86aa49b (first draft for buffer wrapper)

#pragma once

#include <cstddef>
#include <tuple>

#include "definitions.hpp"

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
