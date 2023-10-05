// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
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

#include <numeric>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Allgather.
///
/// This wrapper for \c MPI_Allgather collects the same amount of data from each rank to all ranks. It is semantically
/// equivalent to performing a \c gather() followed by a broadcast of the collected data.
///
/// The following arguments are required:
/// - kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at
/// each rank. See TODO gather_v if the amounts differ.
///
/// The following buffers are optional:
/// - kamping::send_counts() specifying how many elements are sent. This parameter has to be an integer. If
/// omitted, the size of the send buffer is used.
///
/// - kamping::recv_counts() specifying how many elements are received. This parameter has to be an integer. If
/// omitted, the value of send_counts will be used.
///
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
/// all data from all send buffers. The buffer will be resized according to the buffer's
/// kamping::BufferResizePolicy. If this is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to hold all received elements. This requires a size of at least `recv_counts *
/// communicator size`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgather(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(send_counts, recv_counts, recv_buf)
    );

    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;
    KASSERT(
        is_same_on_all_ranks(send_buf.size()),
        "All PEs have to send the same number of elements. Use allgatherv, if you want to send a different number of "
        "elements.",
        assert::light_communication
    );
    // Get the send counts
    using default_send_count_type = decltype(kamping::send_counts_out(alloc_new<int>));
    auto&& send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_count_type>(
            std::tuple(),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_count)>::is_single_element,
        "send_counts() parameter must be a single value."
    );
    constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
    if constexpr (do_compute_send_count) {
        (*send_count.data()) = asserting_cast<int>(send_buf.size());
    }

    // Get the receive counts
    using default_recv_count_type = decltype(kamping::recv_counts_out(alloc_new<int>));
    auto&& recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(recv_count)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );
    constexpr bool do_compute_recv_count = internal::has_to_be_computed<decltype(recv_count)>;
    if constexpr (do_compute_recv_count) {
        (*recv_count.data()) = send_count.get_single_element();
    }
    // TODO remove/adapt this kassert once custom mpi send/recv types are supported
    KASSERT(
        send_count.get_single_element() == recv_count.get_single_element(),
        "Send and recv counts must be equal.",
        assert::light
    );
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));

    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    auto mpi_send_type = mpi_datatype<send_value_type>();
    auto mpi_recv_type = mpi_datatype<recv_value_type>();
    KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(recv_count.get_single_element()) * size();
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgather(
        send_buf.data(),
        asserting_cast<int>(send_count.get_single_element()),
        mpi_send_type,
        recv_buf.data(),
        asserting_cast<int>(recv_count.get_single_element()),
        mpi_recv_type,
        this->mpi_communicator()
    );
    THROW_IF_MPI_ERROR(err, MPI_Allgather);
    return make_mpi_result(std::move(recv_buf), std::move(send_count), std::move(recv_count));
}

/// @brief Wrapper for \c MPI_Allgatherv.
///
/// This wrapper for \c MPI_Allgatherv collects possibly different amounts of data from each rank to all ranks. It is
/// semantically equivalent to performing a \c gatherv() followed by a broadcast of the collected data.
///
/// The following arguments are required:
/// - kamping::send_buf() containing the data that is sent to all other ranks.
///
/// The following parameters are optional but result in communication overhead if omitted:
/// - kamping::recv_counts() containing the number of elements to receive from each rank.
///
/// The following buffers are optional:
/// - kamping::send_counts() specifying how many elements are sent. This parameter has to be an integer. If
/// omitted, the size of the send buffer is used.
///
/// - kamping::recv_buf() containing a buffer for the output.  Afterwards, this buffer will contain
/// all data from all send buffers. The buffer will be resized according to the buffer's
/// kamping::BufferResizePolicy. If resize policy is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to store all received elements. This requires a size of at least  `max(recv_counts[i] +
/// recv_displs[i])` for \c i in `[0, communicator size)`.
///
/// - kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]` elements
/// starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If omitted, this is calculated as the
/// exclusive prefix-sum of `recv_counts`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgatherv(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(send_counts, recv_buf, recv_counts, recv_displs)
    );

    // Get send_buf
    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

    // Get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    // Get the send counts
    using default_send_count_type = decltype(kamping::send_counts_out(alloc_new<int>));
    auto&& send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_count_type>(
            std::tuple(),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_count)>::is_single_element,
        "send_counts() parameter must be a single value."
    );
    constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
    if constexpr (do_compute_send_count) {
        (*send_count.data()) = asserting_cast<int>(send_buf.size());
    }
    // Get the recv counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        );
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");
    // Calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_counts),
        "Receive counts are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_counts) {
        recv_counts.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        this->allgather(
            kamping::send_buf(static_cast<int>(send_count.get_single_element())),
            kamping::recv_buf(recv_counts.get())
        );
    } else {
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
    }

    // Get recv_displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(),
            args...
        );
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));

    // Calculate recv_displs if necessary
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_displs),
        "Receive displacements are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_displs) {
        recv_displs.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_displs.size() >= this->size(), "Recv displs buffer is not large enough.", assert::light);
        std::exclusive_scan(recv_counts.data(), recv_counts.data() + this->size(), recv_displs.data(), 0);
    } else {
        KASSERT(recv_displs.size() >= this->size(), "Recv displs buffer is not large enough.", assert::light);
    }

    auto mpi_send_type = mpi_datatype<send_value_type>();
    auto mpi_recv_type = mpi_datatype<recv_value_type>();
    KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

    auto compute_required_recv_buf_size = [&]() {
        return compute_required_recv_buf_size_in_vectorized_communication(recv_counts, recv_displs, this->size());
    };

    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgatherv(
        send_buf.data(),                                      // sendbuf
        asserting_cast<int>(send_count.get_single_element()), // sendcounts
        mpi_send_type,                                        // sendtype
        recv_buf.data(),                                      // recvbuf
        recv_counts.data(),                                   // recvcounts
        recv_displs.data(),                                   // recvdispls
        mpi_recv_type,                                        // recvtype
        this->mpi_communicator()                              // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Allgatherv);

    return make_mpi_result(std::move(recv_buf), std::move(send_count), std::move(recv_counts), std::move(recv_displs));
}
