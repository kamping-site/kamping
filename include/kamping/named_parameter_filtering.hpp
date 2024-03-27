// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:

#pragma once

#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"

namespace kamping::internal {

/// @brief Wrapper class to store an enum entry (\ref kamping::internal::ParameterType) in a separate type (so that it
/// can be used in a compile time list)
///
/// @tparam ptype ParameterType to store as a type
template <ParameterType ptype>
struct ParameterTypeEntry {
    static constexpr ParameterType parameter_type = ptype; ///< ParameterType to be stored in this type.
};
/// @brief Base template used to concatenate a type to a given std::tuple.
/// based on https://stackoverflow.com/a/18366475
template <typename, typename>
struct PrependType {};

/// @brief Specialization of a class template used to preprend a type to a given std::tuple.
///
/// @tparam Head Type to prepend to the std::tuple.
/// @tparam Tail Types contained in the std::tuple.
template <typename Head, typename... Tail>
struct PrependType<Head, std::tuple<Tail...>> {
    using type = std::tuple<Head, Tail...>; ///< tuple with prepended Head type.
};

/// @brief Base template used to filter a list of types and only keep those whose types meet specified criteria.
/// See the following specialisations for more information.
template <typename...>
struct FilterOut;

/// @brief Specialisation of template class used to filter a list of types and only keep the those whose types meet
/// the specified criteria.
template <typename Predicate>
struct FilterOut<Predicate> {
    using type = std::tuple<>; ///< Tuple of types meeting the specified criteria.
};

/// @brief Specialization of template class used to filter a list of (buffer-)types and discard those for which
/// The template is recursively instantiated to check one type after the other and "insert" it into a
/// std::tuple if it meets the criteria .
/// based on https://stackoverflow.com/a/18366475
///
/// @tparam Predicate Predicate function which has a constexpr static member function `discard()` taking \tparam Head as
/// template parameter and returning a bool indiciating whether head shall be discard (filtered out).
/// @tparam Head Type for which it is checked whether it meets the predicate.
/// @tparam Tail Types that are checked later on during the recursive instantiation.
template <typename Predicate, typename Head, typename... Tail>
struct FilterOut<Predicate, Head, Tail...> {
    using non_ref_first = std::remove_reference_t<Head>; ///< Remove potential reference from Head.
    static constexpr bool discard_elem =
        Predicate::template discard<non_ref_first>();     ///< Predicate which Head has to fulfill to be kept.
    static constexpr auto ptype = parameter_type_v<Head>; ///< ParameterType stored as a static variable in Head.
    using type                  = std::conditional_t<
        discard_elem,
        typename FilterOut<Predicate, Tail...>::type,
        typename PrependType<
            std::integral_constant<parameter_type_t<Head>, ptype>,
            typename FilterOut<Predicate, Tail...>::type>::type>; ///< A std::tuple<T1, ..., Tn> where T1, ..., Tn are
                                                                  ///< those types among Head, Tail... which fulfill the
                                                                  ///< predicate.
};

/// @brief Specialisation of template class for types stored in a std::tuple<...> that is used to filter these types and
/// only keep those which meet certain criteria (see above).
///
/// @tparam Types Types to check.
template <typename Predicate, typename... Types>
struct FilterOut<Predicate, std::tuple<Types...>> {
    using type =
        typename FilterOut<Predicate, Types...>::type; ///< A std::tuple<T1, ..., Tn> where T1, ..., Tn are those
                                                       ///< types among Types... which match the criteria.
};

/// @brief Retrieve the buffer with requested ParameterType from the std::tuple containg all buffers.
///
/// @tparam ptype ParameterType of the buffer to retrieve.
/// @tparam Buffers Types of the data buffers.
/// @param buffers Data buffers out of which the one with requested parameter type is retrieved.
/// @return Reference to the buffer which the requested ParameterType.
template <internal::ParameterType ptype, typename... Buffers>
auto& retrieve_buffer(std::tuple<Buffers...>& buffers) {
    return internal::select_parameter_type_in_tuple<ptype>(buffers);
}

/// @brief Construct tuple containing all buffers specified in \tparam ParamterTypeTuple.
///
/// @tparam ParameterTypeTuple Tuple containing ParameterTypeEntries specifing the entries to be
/// retrieved from the BufferTuple buffers.
/// @tparam Buffers Types of the data buffers.
/// @tparam i Integer sequence.
/// @param buffers Data buffers out of which the ones with parameter types contained in ParameterTypeTuple are
/// retrieved.
/// @return std::tuple containing all requested data buffers.
template <typename ParameterTypeTuple, typename... Buffers, std::size_t... i>
auto construct_buffer_tuple_impl(
    std::tuple<Buffers...>& buffers, std::index_sequence<i...> /*index_sequence*/
) {
    return std::make_tuple(std::move(retrieve_buffer<std::tuple_element_t<i, ParameterTypeTuple>::value>(buffers))...);
}

/// @brief Construct tuple containing all buffers specified in \tparam ParamterTypeTuple.
///
/// @tparam ParameterTypeTuple Tuple containing ParameterTypeEntries specifing the entries to be
/// retrieved from the BufferTuple buffers.
/// @tparam Buffers Types of the data buffers.
/// @param buffers Data buffers out of which the ones with parameter types contained in ParameterTypeTuple are
/// retrieved.
/// @return std::tuple containing all requested data buffers.
template <typename ParameterTypeTuple, typename... Buffers>
auto construct_buffer_tuple(Buffers&&... buffers) {
    // number of buffers that will be contained in the result object (including the receive buffer if it is an out
    // parameter)
    constexpr std::size_t num_output_parameters = std::tuple_size_v<ParameterTypeTuple>;
    auto                  buffers_tuple         = std::tie(buffers...);

    return construct_buffer_tuple_impl<ParameterTypeTuple>(
        buffers_tuple,
        std::make_index_sequence<num_output_parameters>{}
    );
}
} // namespace kamping::internal
