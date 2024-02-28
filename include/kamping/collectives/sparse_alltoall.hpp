// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/ibarrier.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/p2p/iprobe.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/request_pool.hpp"
#include "kamping/result.hpp"

namespace kamping {

/// @brief Class encapsulating a probed message that is ready to be received in a sparse alltoall exchange.
template <typename T, typename Communicator>
class ProbedMessage {
public:
    /// @brief Constructor of a probed message.
    ProbedMessage(Status&& status, Communicator const& comm) : _status(std::move(status)), _comm(comm) {}

    /// @brief Actually receive the probed message into a contiguous memory either provided by the user or allocated by
    /// the library.
    template <typename recv_value_type_tparam = T, typename... Args>
    auto recv(Args... args) const {
        KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(recv_buf, recv_type));

        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<
                                       typename Communicator::template default_container_type<recv_value_type_tparam>>)
            );
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<Communicator::template default_container_type>();
        using recv_buf_type = std::remove_reference_t<decltype(recv_buf)>;

        using recv_value_type   = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
        auto&& recv_type        = internal::determine_mpi_recv_datatype<recv_value_type, decltype(recv_buf)>(args...);
        auto   repack_recv_type = [&]() {
            // we cannot simply forward recv_type as kamping::recv_type as there are checks within recv() depending on
            // whether recv_type is caller provided or not
            if constexpr (internal::has_to_be_computed<decltype(recv_type)>) {
                return kamping::recv_type_out(recv_type.underlying());
            } else {
                return kamping::recv_type(recv_type.underlying());
            }
        };
        _comm.recv(
            kamping::recv_buf<recv_buf_type::resize_policy>(recv_buf.underlying()),
            repack_recv_type(),
            kamping::recv_count(recv_count_signed(recv_type.underlying())),
            kamping::source(_status.source_signed()),
            tag(_status.tag())
        );

        return internal::make_mpi_result<std::tuple<Args...>>(std::move(recv_buf), std::move(recv_type));
    }

    /// @brief Computes the size of the probed message depending on the used datatype.
    int recv_count_signed(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        if (datatype == MPI_DATATYPE_NULL) {
            datatype = mpi_datatype<T>();
        }
        return _status.count_signed(datatype);
    }

    /// @brief Computes the size of the probed message depending on the used datatype.
    size_t recv_count(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        return asserting_cast<size_t>(recv_count_signed(datatype));
    }

    /// @brief Returns the source of the probed message.
    int source_signed() const {
        return _status.source_signed();
    }

    /// @brief Returns the source of the probed message.
    size_t source() const {
        return _status.source();
    }

private:
    Status              _status;
    Communicator const& _comm;
};
} // namespace kamping
namespace kamping::experimental {
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

/// @brief List of parameter type (entries) which should not be included in the result object.
using parameter_types_to_ignore_for_result_object = internal::type_list<
    std::integral_constant<internal::ParameterType, internal::ParameterType::sparse_send_buf>,
    std::integral_constant<internal::ParameterType, internal::ParameterType::on_message>,
    std::integral_constant<internal::ParameterType, internal::ParameterType::destination>>;

/// @brief Determines whether a given buffer with \tparam BufferType should we included in the result object.
///
/// @tparam BufferType Type of the data buffer.
/// @return \c True iff the \tparam BufferType has the static bool members \c is_owning and \c is_out_buffer and both
/// values are true.
template <typename BufferType>
constexpr bool keep_entry() {
    using ptype_entry = std::integral_constant<internal::ParameterType, BufferType::parameter_type>;
    return !experimental::parameter_types_to_ignore_for_result_object::contains<ptype_entry>;
}
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
template <typename Predicate, typename Head, typename... Tail>
struct FilterOut<Predicate, Head, Tail...> {
    using non_ref_first = std::remove_reference_t<Head>; ///< Remove potential reference from Head.
    static constexpr bool discard_elem =
        Predicate::template discard<non_ref_first>(); ///< Predicate which Head has to fulfill to be kept.
    static constexpr internal::ParameterType ptype =
        non_ref_first::parameter_type; ///< ParameterType stored as a static variable in Head.
    using type = std::conditional_t<
        discard_elem,
        typename FilterOut<Predicate, Tail...>::type,
        typename PrependType<
            std::integral_constant<internal::ParameterType, ptype>,
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

/// @brief Retrieve the Buffer with given ParameterType from the tuple Buffers.
///
/// @tparam ParameterTypeTuple Tuple containing std::integral_constant<ParameterType> specifing the entries to be
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
        std::move(experimental::retrieve_buffer<std::tuple_element_t<i, ParameterTypeTuple>::value>(buffers))...
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

    return experimental::construct_buffer_tuple_for_result_object_impl<ParameterTypeTuple>(
        buffers_tuple,
        std::make_index_sequence<num_output_parameters>{}
    );
}
} // namespace kamping::experimental

///@brief Predicate to check whether an argument provided to sparse_alltoall shall be discard in the send call.
struct PredicateForSparseAlltoall {
    ///@brief Discard functions to check whether an argument provided to sparse_alltoall shall be discard in the send
    ///call.
    ///
    ///@tparam Arg Argument to be checked.
    ///@return \c True (i.e. discard) iff Arg's parameter_type is `sparse_send_buf`, `on_message` or `destination`.
    template <typename Arg>
    static constexpr bool discard() {
        using namespace kamping::internal;
        using ptypes_to_ignore = type_list<
            std::integral_constant<ParameterType, ParameterType::sparse_send_buf>,
            std::integral_constant<ParameterType, ParameterType::on_message>,
            std::integral_constant<ParameterType, ParameterType::destination>>;
        using ptype_entry = std::integral_constant<ParameterType, Arg::parameter_type>;
        return ptypes_to_ignore::contains<ptype_entry>;
    }
};
template <typename... Args>
auto filter(Args&&... args) {
    using ArgsToKeep = typename kamping::experimental::FilterOut<PredicateForSparseAlltoall, std::tuple<Args...>>::type;
    return kamping::experimental::construct_buffer_tuple_for_result_object<ArgsToKeep>(args...);
}

/// @brief Sparse alltoall exchange using the NBX algorithm(Hoefler et al., "Scalable communication protocols for
/// dynamic sparse data", ACM Sigplan Noctices 45.5, 2010.)
///
/// This function provides a sparse interface for personalized all-to-all communication using
/// direct message exchange and thus achieving linear complexity in the number of messages to be sent (in
/// contrast to \c MPI_Alltoallv which exhibits complexity (at least) linear in the size of the communicator due to its
/// interface). To achieve this time complexity we can no longer rely on an array of size of the communicator for send
/// counts. Instead we use a sparse representation of the data to be sent.
///
/// The following parameters are required:
/// - \ref kamping::sparse_send_buf() containing the messages to be sent to other ranks. Differently from plain
/// alltoallv, in alltoallv_sparse \c send_buf() encapsulates a container consisting of destination-message pairs. Each
/// such pair has to be decomposable via structured bindings with the first parameter being convertible to int and the
/// second parameter being the actual message to be sent for which we require the usual send_buf properties (i.e.,
/// `data()` and `size()` member function and the exposure of a `value_type`)). Messages of size 0 are not sent.
/// - \ref kamping::on_message() containing a callback function `cb` which is responsible to process the received
/// messages via a \ref kamping::ProbedMessage object. The callback function `cb` gets called for each probed message
/// ready to be received via `cb(probed_message)`. See \ref kamping::ProbedMessage for the member functions to be called
/// on the object.
///
/// The following buffers are optional:
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on each message's underlying \c value_type.
/// - \ref kamping::tag() the tag added to the directly exchanged messages. Defaults to the communicator's default tag
/// (\ref Communicator::default_tag()) if not present.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional parameters described above.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
void kamping::Communicator<DefaultContainerType, Plugins...>::alltoallv_sparse(Args... args) const {
    // Get all parameter objects
    using SelfType = kamping::Communicator<DefaultContainerType, Plugins...>;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(sparse_send_buf, on_message),
        KAMPING_OPTIONAL_PARAMETERS(send_type, tag)
    );
    int tag = 0;

    // Get send_buf
    auto const& dst_message_container =
        internal::select_parameter_type<internal::ParameterType::sparse_send_buf>(args...);
    using dst_message_container_type =
        typename std::remove_reference_t<decltype(dst_message_container.underlying())>::value_type;
    using message_type = typename std::tuple_element_t<1, dst_message_container_type>;
    // support message_type being a single element.
    using message_value_type =
        typename internal::ValueTypeWrapper<internal::has_data_member_v<message_type>, message_type>::value_type;

    // Get callback
    auto const& on_message_cb = internal::select_parameter_type<internal::ParameterType::on_message>(args...);

    RequestPool<DefaultContainerType> request_pool;
    for (auto const& [dst, msg]: dst_message_container.underlying()) {
        auto send_buf = kamping::send_buf(msg);

        if (send_buf.size() > 0) {
            int       dst_       = dst; // cannot capture structured binding variable
            int const send_count = asserting_cast<int>(send_buf.size());
            auto      callable   = [&](auto... argsargs) {
                issend(
                    std::move(send_buf),
                    kamping::send_count(send_count),
                    destination(dst_),
                    request(request_pool.get_request()),
                    std::move(argsargs)...
                );
            };
            std::apply(callable, filter(args...));
        }
    }

    Status  status;
    Request barrier_request(MPI_REQUEST_NULL);
    while (true) {
        bool const got_message = iprobe(kamping::tag(tag), kamping::status_out(status));
        if (got_message) {
            ProbedMessage<message_value_type, SelfType> probed_message{std::move(status), *this};
            on_message_cb.underlying()(probed_message);
        }
        if (!barrier_request.is_null()) {
            if (barrier_request.test()) {
                break;
            }
        } else {
            if (request_pool.test_all()) {
                ibarrier(request(barrier_request));
            }
        }
    }
    barrier();
}
