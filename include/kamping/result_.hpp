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
// <https://www.gnu.org/licenses/>.:

#pragma once

/// @file
/// @brief Some functions and types simplifying/enabling the development of wrapped \c MPI calls in KaMPIng.

#include <iostream>
#include <optional>
#include <tuple>
#include <utility>

#include "kamping/has_member.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "named_parameter_selection.hpp"

namespace kamping {

/// @brief MPIResult contains the result of a \c MPI call wrapped by KaMPIng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transferred to KaMPIng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every
/// wrapped \c MPI call. If a specific call does not use a buffer category, you
/// have to provide ResultCategoryNotUsed instead.
///
/// @tparam Args Types of return data buffers.
template <typename... Args>
class MPIResult_ {
public:
    /// @brief \c true, if the result does not encapsulate any data.
    static constexpr bool is_empty = (sizeof...(Args) == 0);
    /// @brief \c true, if the result encapsulates a recv_buf.
    static constexpr bool has_recv_buffer = internal::has_parameter_type<internal::ParameterType::recv_buf, Args...>();
    /// @brief \c true, if the result encapsulates a send_recv_buf.
    static constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    static_assert(
        !(has_recv_buffer && has_send_recv_buffer),
        "We cannot have a recv and a send_recv buffer contained in the result object."
    );

    /// @brief Constructor for MPIResult.
    ///
    /// @param data std::tuple containing all data buffers to be returned.
    MPIResult_(std::tuple<Args...>&& data) : _data(std::move(data)) {}

    /// @brief Extracts the \c kamping::Status from the MPIResult object.
    ///
    /// This function is only available if the underlying status is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying status object.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::status, T>(), bool> = true>
    decltype(auto) extract_status() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::status>(_data).extract();
    }

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_buf, T>(), bool> = true>
    decltype(auto) extract_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_buf>(_data).extract();
    }

    /// @brief Extracts the \c send_recv_buffer from the MPIResult object. @todo discuss this
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_buf, T>(), bool> =
            true>
    decltype(auto) extract_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_buf>(_data).extract();
    }

    /// @brief Extracts the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the receive counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_counts, T>(), bool> = true>
    decltype(auto) extract_recv_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_counts>(_data).extract();
    }

    /// @brief Extracts the \c recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the recv count.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_count, T>(), bool> = true>
    decltype(auto) extract_recv_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_count>(_data).extract();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_displs, T>(), bool> = true>
    decltype(auto) extract_recv_displs() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_displs>(_data).extract();
    }

    /// @brief Extracts the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_counts, T>(), bool> = true>
    decltype(auto) extract_send_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_counts>(_data).extract();
    }

    /// @brief Extracts the \c send_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_count, T>(), bool> = true>
    decltype(auto) extract_send_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_count>(_data).extract();
    }

    /// @brief Extracts the \c send_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_displs, T>(), bool> = true>
    decltype(auto) extract_send_displs() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_displs>(_data).extract();
    }

    /// @brief Extracts the \c send_recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_recv_count.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_count, T>(), bool> =
            true>
    decltype(auto) extract_send_recv_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_count>(_data).extract();
    }

    /// @brief Extracts the \c send_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_type, T>(), bool> = true>
    decltype(auto) extract_send_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_type>(_data).extract();
    }

    /// @brief Extracts the \c recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_type, T>(), bool> = true>
    decltype(auto) extract_recv_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_type>(_data).extract();
    }

    /// @brief Extracts the \c send_recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_type, T>(), bool> =
            true>
    decltype(auto) extract_send_recv_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_type>(_data).extract();
    }

    /// @brief Get the underlying data from the i-th buffer in the result object. This method is part of the
    /// structured binding enabling machinery.
    ///
    /// @tparam i Index of the data buffer to extract.
    /// @return Returns a reference to the underlying data of the i-th data buffer.
    template <std::size_t i>
    auto& get() {
        return std::get<i>(_data).underlying();
    }

    /// @brief Get the underlying data from the i-th buffer in the result object. This method is part of the
    /// structured binding enabling machinery.
    ///
    /// @tparam i Index of the data buffer to extract.
    /// @return Returns a reference to the underlying data of the i-th data buffer.
    template <std::size_t i>
    auto const& get() const {
        return std::get<i>(_data).underlying();
    }

private:
    std::tuple<Args...> _data; ///< tuple storing the data buffers
};

/// @brief Primary template for result trait indicates whether the result object is empty.
template <typename T>
constexpr bool is_result_empty_v = false;

/// @brief Template specialization for result trait indicates whether the result object is
/// empty.
template <typename... Args>
constexpr bool is_result_empty_v<MPIResult_<Args...>> = MPIResult_<Args...>::is_empty;

/// @brief Template specialization for result trait indicates whether the result object is
/// empty.
template <>
inline constexpr bool is_result_empty_v<void> = true;

} // namespace kamping

namespace std {

/// @brief Specialization of the std::tuple_size for \ref kamping::MPIResult_. Part of the structured binding machinery.
///
/// @tparam Args Automatically deducted template parameters.
template <typename... Args>
struct tuple_size<kamping::MPIResult_<Args...>> {
    static constexpr size_t value = sizeof...(Args); ///< Number of data buffers in the \ref kamping::MPIResult_.
};

/// @brief Specialization of the std::tuple_element for \ref kamping::MPIResult_. Part of the structured binding
/// machinery.
///
/// @param index Index of the entry of \ref kamping::MPIResult_ for which the underlying data type shall be deduced.
/// @tparam Args Automatically deducted template parameters.
template <size_t index, typename... Args>
struct tuple_element<index, kamping::MPIResult_<Args...>> {
    using type = std::remove_reference_t<decltype(declval<kamping::MPIResult_<Args...>>().template get<index>()
    )>; ///< Type of the underlying data of the i-th data buffer in the result object.
};

} // namespace std

namespace kamping::internal {

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

/// @brief Wrapper class to store an enum entry (\ref kamping::internal::ParameterType) in a separate type (so that it
/// can be used in a compile time list)
///
/// @tparam ptype ParameterType to store as a type
template <ParameterType ptype>
struct ParameterTypeEntry {
    static constexpr ParameterType parameter_type = ptype; ///< ParameterType to be stored in this type.
};

/// @brief List of parameter type (entries) which should not be included in the result object.
using parameter_types_to_ignore_for_result_object = type_list<
    ParameterTypeEntry<ParameterType::op>,
    ParameterTypeEntry<ParameterType::source>,
    ParameterTypeEntry<ParameterType::destination>,
    ParameterTypeEntry<ParameterType::statuses>,
    ParameterTypeEntry<ParameterType::request>,
    ParameterTypeEntry<ParameterType::root>,
    ParameterTypeEntry<ParameterType::tag>,
    ParameterTypeEntry<ParameterType::send_mode>,
    ParameterTypeEntry<ParameterType::values_on_rank_0>>;

/// @brief Determines whether a given buffer with \tparam BufferType should we included in the result object.
///
/// @tparam BufferType Type of the data buffer.
/// @return \c True iff the \tparam BufferType has the static bool members \c is_owning and \c is_out_buffer and both
/// values are true.
template <typename BufferType>
constexpr bool is_returnable_owning_out_data_buffer() {
    using ptype_entry = ParameterTypeEntry<BufferType::parameter_type>;
    if constexpr (parameter_types_to_ignore_for_result_object::contains<ptype_entry>) {
        return false;
    } else {
        return BufferType::is_owning && BufferType::is_out_buffer; ///< Predicate which Head has to fulfill to be kept.
    }
}

/// @brief Base template used to filter a list of types and only keep those whose types meet specified criteria.
/// See the following specialisations for more information.
template <typename...>
struct FilterOwningOut;

/// @brief Specialisation of template class used to filter a list of types and only keep the those whose types meet
/// the specified criteria.
template <>
struct FilterOwningOut<> {
    using type = std::tuple<>; ///< Tuple of types meeting the specified criteria.
};

/// @brief Specialization of template class used to filter a list of (buffer-)types and only keep those whose types meet
/// the following criteria:
/// - an object of the type owns its underlying storage
/// - an object of the type is an out buffer
/// - @see \ref is_returnable_owning_out_data_buffer()
///
/// The template is recursively instantiated to check one type after the other and "insert" it into a
/// std::tuple if it meets the criteria.
/// based on https://stackoverflow.com/a/18366475
///
/// @tparam Head Type for which it is checked whether it meets the predicate.
/// @tparam Tail Types that are checked later on during the recursive instantiation.
template <typename Head, typename... Tail>
struct FilterOwningOut<Head, Tail...> {
    using non_ref_first = std::remove_reference_t<Head>; ///< Remove potential reference from Head.
    static constexpr bool predicate =
        is_returnable_owning_out_data_buffer<non_ref_first>(); ///< Predicate which Head has to fulfill to be kept.
    static constexpr ParameterType ptype =
        non_ref_first::parameter_type; ///< ParameterType stored as a static variable in Head.
    using type = std::conditional_t<
        predicate,
        typename PrependType<ParameterTypeEntry<ptype>, typename FilterOwningOut<Tail...>::type>::type,
        typename FilterOwningOut<Tail...>::type>; ///< A std::tuple<T1, ..., Tn> where T1, ..., Tn are those types among
                                                  ///< Head, Tail... which fulfill the predicate.
};

/// @brief Specialisation of template class for types stored in a std::tuple<...> that is used to filter these types and
/// only keep those which meet certain criteria (see above).
///
/// @tparam Types Types to check.
template <typename... Types>
struct FilterOwningOut<std::tuple<Types...>> {
    using type = typename FilterOwningOut<Types...>::type; ///< A std::tuple<T1, ..., Tn> where T1, ..., Tn are those
                                                           ///< types among Types... which match the criteria.
};

/// @brief Template class to prepend the ParameterTypeEntry<ParameterType::ptype> type to a given std::tuple.
/// @tparam ptype ParameterType to prepend
/// @tparam Tuple An std::tuple.
template <ParameterType ptype, typename Tuple>
struct PrependParameterType {
    using type = typename PrependType<ParameterTypeEntry<ptype>, Tuple>::
        type; ///< Concatenated tuple, i.e. type = std::tuple<TypeToPrepend, (Type contained in Tuple)... >.
};

/// @brief Retrieve the buffer with requested ParameterType from the std::tuple containg all buffers.
///
/// @tparam ptype ParameterType of the buffer to retrieve.
/// @tparam Buffers Types of the data buffers.
/// @param buffers Data buffers out of which the one with requested parameter type is retrieved.
/// @return Reference to the buffer which the requested ParameterType.
template <ParameterType ptype, typename... Buffers>
auto& retrieve_buffer(std::tuple<Buffers...>& buffers) {
    return select_parameter_type_in_tuple<ptype>(buffers);
}

/// @brief Retrieve the Buffer with given ParameterType from the tuple Buffers.
///
/// @tparam ParameterTypeTuple Tuple containing specialized ParameterTypeEntry types specifing the entries to be
/// retrieved from the BufferTuple buffers.
/// @tparam Buffers Types of the data buffers.
/// @tparam i Integer sequence.
/// @param buffers Data buffers out of which the ones with parameter types contained in ParameterTypeTuple are
/// retrieved.
/// @return std::tuple containing all requested data buffers.
template <typename ParameterTypeTuple, typename... Buffers, std::size_t... i>
auto construct_buffer_tuple_for_result_object_impl(
    std::tuple<Buffers...>& buffers, std::index_sequence<i...> /*index_sequence*/
) {
    return std::make_tuple(
        std::move(retrieve_buffer<std::tuple_element_t<i, ParameterTypeTuple>::parameter_type>(buffers))...
    );
}

/// @brief Retrieve the Buffer with given ParameterType from the tuple Buffers.
///
/// @tparam ParameterTypeTuple Tuple containing specialized ParameterTypeEntry types specifing the entries to be
/// retrieved from the BufferTuple buffers.
/// @tparam Buffers Types of the data buffers.
/// @param buffers Data buffers out of which the ones with parameter types contained in ParameterTypeTuple are
/// retrieved.
/// @return std::tuple containing all requested data buffers.
template <typename ParameterTypeTuple, typename... Buffers>
auto construct_buffer_tuple_for_result_object(Buffers&&... buffers) {
    // number of buffers that will be contained in the result object (including the receive buffer if it is an out
    // parameter)
    constexpr std::size_t num_output_parameters = std::tuple_size_v<ParameterTypeTuple>;
    auto                  buffers_tuple         = std::tie(buffers...);

    return construct_buffer_tuple_for_result_object_impl<ParameterTypeTuple>(
        buffers_tuple,
        std::make_index_sequence<num_output_parameters>{}
    );
}

/// @brief Determines whether only the recv (send_recv) buffer or multiple different buffers will be returned.
/// @tparam CallerProvidedOwningOutBuffers An std::tuple containing the types of the owning, out buffers explicitly
/// requested by the caller of the wrapped MPI call.
/// @returns \c True if the recv (send_recv) buffer is either not mentioned explicitly and no other (owning) out buffers
/// are requested or the only explicitly requested owning out buffer is the recv_buf. \c False otherwise.
template <typename CallerProvidedOwningOutBuffers>
constexpr bool return_recv_or_send_recv_buffer_only() {
    constexpr std::size_t num_caller_provided_owning_out_buffers = std::tuple_size_v<CallerProvidedOwningOutBuffers>;
    if constexpr (num_caller_provided_owning_out_buffers == 0) {
        return true;
    } else if constexpr (num_caller_provided_owning_out_buffers == 1 && std::tuple_element_t<0, CallerProvidedOwningOutBuffers>::parameter_type == ParameterType::recv_buf) {
        return true;
    } else if constexpr (num_caller_provided_owning_out_buffers == 1 && std::tuple_element_t<0, CallerProvidedOwningOutBuffers>::parameter_type == ParameterType::send_recv_buf) {
        return true;
    } else {
        return false;
    }
}

/// @brief Checks whether a buffer with parameter type recv_buf or a buffer with type send_recv_buf is present and
/// returns the found parameter type. Note that we require that either a recv_buf or a send_recv_buf is present.
///
/// @tparam Buffers All buffer types to be searched for type `recv_buf` or `send_recv_buf`.
/// @returns The parameter type of the first buffer whose parameter type is recv_buf or send_recv_buf.
template <typename... Buffers>
constexpr ParameterType determine_recv_buffer_type() {
    constexpr bool has_recv_buffer = internal::has_parameter_type<internal::ParameterType::recv_buf, Buffers...>();
    constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Buffers...>();
    static_assert(has_recv_buffer ^ has_send_recv_buffer, "either a recv or a send_recv buffer must be present");
    if constexpr (has_recv_buffer) {
        return ParameterType::recv_buf;
    } else {
        return ParameterType::send_recv_buf;
    }
}

/// @brief Returns True iff only a status buffer but no recv or send_recv buffer is present.
/// Note that we require that a status (see e.g. \ref kamping::Communicator::probe()or \ref
/// kamping::Communicator::iprobe()) or either a recv_buf or a send_recv_buf is present.
///
/// @tparam Buffers All buffer types to be searched for type `status`.
template <typename... Buffers>
constexpr bool has_status_but_no_recv_or_send_recv_buf() {
    constexpr bool has_status_buffer = internal::has_parameter_type<internal::ParameterType::status, Buffers...>();
    constexpr bool has_recv_buffer   = internal::has_parameter_type<internal::ParameterType::recv_buf, Buffers...>();
    constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Buffers...>();
    static_assert(
        has_status_buffer || (has_recv_buffer ^ has_send_recv_buffer),
        "a status or either a recv or a send_recv buffer must be present"
    );
    if constexpr (has_recv_buffer || has_send_recv_buffer) {
        return false;
    } else {
        return has_status_buffer;
    }
}

/// @brief Construct result object for a wrapped MPI call. Four different cases are handled:
/// (Note that in the following recv_buffer also means send_recv_buffer for functions such as MPI_Bcast etc.)
/// a) The recv_buffer owns its underlying data (i.e. the received data has to be returned via the result object):
///
/// a.1) The recv_buffer is the only buffer to be returned, i.e. the only caller provided owning out buffer:
/// In this case, the recv_buffers's underlying data is extracted and returned directly (by value).
///
/// a.2) There are multiple buffers to be returned and recv_buffer is explicitly provided by the caller:
/// In this case a \ref kamping::MPIResult_ object is created, which stores the buffers to return (owning out buffers)
/// in a std::tuple respecting the order in which these buffers where provided to the wrapped MPI call. This enables
/// unpacking the object via structured binding.
///
/// a.3) There are more data buffers to be returned and recv_buffer is *not* explicitly provided by the caller:
/// In this case a \ref kamping::MPIResult_ object is created, which stores the buffers to return. The
/// recv_buffer is always the first entry in the result object followed by the other buffers respecting the order in
/// which these buffers where provided to the wrapped MPI call.
///
/// b) The recv_buffer only references its underlying data (i.e. it is a non-owinig out buffer):
/// In this case recv_buffer is not part of the result object. The \ref kamping::MPIResult_ object stores the buffer to
/// return (owning buffers for which a *_out() named parameter was passed to the wrapped MPI call) in a std::tuple
/// respecting the order in which these buffers where provided to the wrapped MPI call.
///
/// @tparam CallerProvidedArgs Types of arguments passed to the wrapped MPI call.
/// @tparam Buffers Types of data buffers created/filled within the wrapped MPI call.
/// @param buffers data buffers created/filled within the wrapped MPI call.
/// @return result object as specified above.
///
/// @see \ref docs/named_parameters.md
template <typename CallerProvidedArgs, typename... Buffers>
auto make_mpi_result_(Buffers&&... buffers) {
    // filter named parameters provided to the wrapped MPI function and keep only owning out parameters (=owning out
    // buffers)
    using CallerProvidedOwningOutParameters = typename internal::FilterOwningOut<CallerProvidedArgs>::type;
    constexpr std::size_t num_caller_provided_owning_out_buffers = std::tuple_size_v<CallerProvidedOwningOutParameters>;
    if constexpr (has_status_but_no_recv_or_send_recv_buf<Buffers...>()) {
        // do no special handling for receive buffer at all, since there is none.
        return MPIResult_(construct_buffer_tuple_for_result_object<CallerProvidedOwningOutParameters>(buffers...));
    } else {
        // receive (send-receive) buffer needs (potentially) a special treatment (if it is an owning (out) buffer and
        // provided by the caller)
        constexpr internal::ParameterType recv_parameter_type = determine_recv_buffer_type<Buffers...>();
        auto&          recv_or_send_recv_buffer = internal::select_parameter_type<recv_parameter_type>(buffers...);
        constexpr bool recv_or_send_recv_buf_is_owning =
            std::remove_reference_t<decltype(recv_or_send_recv_buffer)>::is_owning;
        constexpr bool recv_or_send_recv_buffer_is_owning_and_provided_by_caller =
            has_parameter_type_in_tuple<recv_parameter_type, CallerProvidedOwningOutParameters>();

        // special case 1: recv (send_recv) buffer is not owning
        if constexpr (!recv_or_send_recv_buf_is_owning) {
            if constexpr (num_caller_provided_owning_out_buffers == 0) {
                // there are no buffers to return
                return;
            } else {
                // no special treatement of recv buffer is needed as the recv_buffer is not part of the result
                // object anyway.
                return MPIResult_(construct_buffer_tuple_for_result_object<CallerProvidedOwningOutParameters>(buffers...
                ));
            }
        }
        // specialcase 2: recv (send_recv) buffer is the only owning out parameter
        else if constexpr (return_recv_or_send_recv_buffer_only<CallerProvidedOwningOutParameters>()) {
            // if only the receive buffer shall be returned, its underlying data is returned directly instead of a
            // wrapping result object
            return recv_or_send_recv_buffer.extract();
        }

        // case A: recv (send_recv) buffer is provided by caller (and owning)
        else if constexpr (recv_or_send_recv_buffer_is_owning_and_provided_by_caller) {
            return MPIResult_(construct_buffer_tuple_for_result_object<CallerProvidedOwningOutParameters>(buffers...));
        }
        // case B: recv buffer is not provided by caller -> recv buffer will be stored as first entry in
        // underlying result object
        else {
            using ParametersToReturn =
                typename PrependParameterType<recv_parameter_type, CallerProvidedOwningOutParameters>::type;
            return MPIResult_(construct_buffer_tuple_for_result_object<ParametersToReturn>(buffers...));
        }
    }
}

} // namespace kamping::internal
