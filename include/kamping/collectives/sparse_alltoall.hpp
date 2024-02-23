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
    ProbedMessage(MPI_Status&& status, Communicator const& comm) : _status(std::move(status)), _comm(comm) {}

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

    int recv_count_signed(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        if (datatype == MPI_DATATYPE_NULL) {
            datatype = mpi_datatype<T>();
        }
        return _status.count_signed(datatype);
    }

    size_t recv_count(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        return asserting_cast<size_t>(recv_count_signed(datatype));
    }

    int source_signed() const {
        return _status.source_signed();
    }

    size_t source() const {
        return _status.source();
    }

private:
    Status              _status;
    Communicator const& _comm;
};
} // namespace kamping

/// @brief Sparse alltoall exchange using the NBX algorithm by Hoeffler et al..
///
/// This wrapper for \c MPI_Alltoallv sends different amounts of data from some/all ranks to some other/all ranks using
/// direct message exchange and therefore realizing a complexity linear in the number of messages to be sent. To achieve
/// this time complexity we can no longer rely on an array of size of the communicator for send counts. Instead we use a
/// sparse representation of the data to be sent.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the messages to be sent to other ranks. Differently from plain alltoallv, in
/// alltoallv_sparse \c send_buf() encapsulates a container consisting of destination-message pairs. Each such pair has
/// to be decomposable via structured bindings with the first parameter being convertible to int and the second
/// parameter being the actual message to be sent for which we require the usual send_buf properties (i.e., `data()` and
/// `size()` member function and the exposure of a `value_type`)).
/// - \ref kamping::on_message() containing a callback function `cb` which is responsible to process the received
/// messages via a \ref kamping::ProbedMessage object. The callback function `cb` gets called for each probed message
/// ready to be received via `cb(probed_message)`. See \ref kamping::ProbedMessage for the member functions to be called
/// on the object.
///
/// The following buffers are optional:
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on each message's underlying \c value_type.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional parameters described above.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename Callback, typename... Args>
void kamping::Communicator<DefaultContainerType, Plugins...>::alltoallv_sparse(Callback on_message, Args... args)
    const {
    // Get all parameter objects
    using SelfType = kamping::Communicator<DefaultContainerType, Plugins...>;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(sparse_send_buf), KAMPING_OPTIONAL_PARAMETERS());
    int tag = 0;

    // Get send_buf
    auto const& dst_message_container =
        internal::select_parameter_type<internal::ParameterType::sparse_send_buf>(args...);
    using dst_message_container_type =
        typename std::remove_reference_t<decltype(dst_message_container.underlying())>::value_type;
    using message_value_type = typename std::tuple_element_t<1, dst_message_container_type>::value_type;

    RequestPool<DefaultContainerType> request_pool;
    for (auto const& [dst, msg]: dst_message_container.underlying()) {
        if (std::size(msg) > 0) {
            issend(kamping::send_buf(msg), destination(dst), kamping::tag(tag), request(request_pool.get_request()));
        }
    }

    MPI_Status status;
    Request    barrier_request(MPI_REQUEST_NULL);
    while (true) {
        bool const got_message = iprobe(kamping::tag(tag), kamping::status_out(status));
        if (got_message) {
            ProbedMessage<message_value_type, SelfType> probed_message{std::move(status), *this};
            on_message(probed_message);
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
