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

#pragma once

#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Alltoall.
///
/// This wrapper for \c MPI_Alltoall sends the same amount of data from each rank to each rank. The following
/// buffers are required:
/// - kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank and divisible by the size of the communicator unless a send_count is explicitly given as parameter. Each
/// rank receives the same number of elements from this buffer.
///
/// The following parameters are optional:
/// - kamping::send_counts() specifying how many elements are sent. This parameter has to be an integer. If
/// omitted, the size of send buffer divided by communicator size is used.
///
/// - kamping::recv_counts() specifying how many elements are received. This parameter has to be an integer. If
/// omitted, the value of send_counts will be used.
///
/// - kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
/// the data received as specified for send_buf. The data received from rank 0 comes first, followed by the data
/// received from rank 1, and so on. The buffer will be resized according to the buffer's
/// kamping::BufferResizePolicy. If this is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to hold all received elements. This requires a size of at least `recv_counts *
/// communicator size`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoall(Args... args) const {
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_counts, recv_counts)
    );

    auto const& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    MPI_Datatype mpi_send_type    = mpi_datatype<send_value_type>();

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

    // Get the send and receive counts
    using default_send_count_type = decltype(kamping::send_counts(alloc_new<int>));
    using default_recv_count_type = decltype(kamping::recv_counts(alloc_new<int>));
    auto&& send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_count_type>(
            std::make_tuple(asserting_cast<int>(send_buf.size() / size())),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_count)>::is_single_element,
        "send_counts() parameter must be a single value."
    );

    auto&& recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::make_tuple(send_count.get_single_element()),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(recv_count)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );

    KASSERT(
        (internal::has_to_be_computed<decltype(send_count)> || send_buf.size() % size() == 0lu),
        "There are no send counts given and the number of elements in send_buf is not divisible by the number of "
        "ranks "
        "in the communicator.",
        assert::light
    );

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(recv_count.get_single_element()) * size();
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // These KASSERTs are required to avoid a false warning from g++ in release mode
    KASSERT(send_buf.data() != nullptr, assert::light);
    KASSERT(recv_buf.data() != nullptr, assert::light);

    [[maybe_unused]] int err = MPI_Alltoall(
        send_buf.data(),
        send_count.get_single_element(),
        mpi_send_type,
        recv_buf.data(),
        recv_count.get_single_element(),
        mpi_recv_type,
        mpi_communicator()
    );

    THROW_IF_MPI_ERROR(err, MPI_Alltoall);
    return make_mpi_result(std::move(recv_buf), std::move(send_count), std::move(recv_count));
}

/// @brief Wrapper for \c MPI_Alltoallv.
///
/// This wrapper for \c MPI_Alltoallv sends the different amounts of data from each rank to each rank. The following
/// buffers are required:
/// - kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at least
/// the sum of the send_counts argument.
///
/// - kamping::send_counts() containing the number of elements to send to each rank.
///
/// The following parameters are optional but result in communication overhead if omitted:
/// - kamping::recv_counts() containing the number of elements to receive from each rank.
///
/// The following buffers are optional:
/// - kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
/// the data received as specified for send_buf. The buffer will be resized according to the buffer's
/// kamping::BufferResizePolicy. If resize policy is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to store all received elements. This requires a size of at least  `max(recv_counts[i] +
/// recv_displs[i])` for \c i in `[0, communicator size)`.
///
/// - kamping::send_displs() containing the offsets of the messages in send_buf. The `send_counts[i]` elements
/// starting at `send_buf[send_displs[i]]` will be sent to rank `i`. If omitted, this is calculated as the exclusive
/// prefix-sum of `send_counts`.
///
/// - kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]` elements
/// starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If omitted, this is calculated as the
/// exclusive prefix-sum of `recv_counts`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer, counts and displacements if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoallv(Args... args) const {
    // Get all parameter objects
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
        KAMPING_OPTIONAL_PARAMETERS(recv_counts, recv_buf, send_displs, recv_displs)
    );

    // Get send_buf
    auto const& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    MPI_Datatype mpi_send_type    = mpi_datatype<send_value_type>();

    // Get send_counts
    auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...);
    using send_counts_type  = typename std::remove_reference_t<decltype(send_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_counts_type>, int>, "Send counts must be of type int");
    static_assert(
        !internal::has_to_be_computed<decltype(send_counts)>,
        "Send counts must be given as an input parameter"
    );
    KASSERT(send_counts.size() >= this->size(), "Send counts buffer is not large enough.", assert::light);

    // Get recv_counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        );
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");

    // Get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    // Get send_displs
    using default_send_displs_type = decltype(kamping::send_displs_out(alloc_new<DefaultContainerType<int>>));
    auto&& send_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
            std::tuple(),
            args...
        );
    using send_displs_type = typename std::remove_reference_t<decltype(send_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_displs_type>, int>, "Send displs must be of type int");

    // Get recv_displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(),
            args...
        );
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    // Check that send and receive buffers have matching types
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match."
    );
    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");
    KASSERT(mpi_send_type == mpi_recv_type, "The MPI receive type does not match the MPI send type.", assert::light);

    // Calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_counts),
        "Receive counts are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_counts) {
        /// @todo make it possible to test whether this additional communication is skipped
        recv_counts.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        this->alltoall(kamping::send_buf(send_counts.get()), kamping::recv_buf(recv_counts.get()));
    } else {
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
    }

    // Calculate send_displs if necessary
    constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_send_displs),
        "Send displacements are given on some ranks and have to be computed on others",
        assert::light_communication
    );

    if constexpr (do_calculate_send_displs) {
        send_displs.resize_if_requested([&]() { return this->size(); });
        KASSERT(send_displs.size() >= this->size(), "Send displs buffer is not large enough.", assert::light);
        std::exclusive_scan(send_counts.data(), send_counts.data() + this->size(), send_displs.data(), 0);
    } else {
        KASSERT(send_displs.size() >= this->size(), "Send displs buffer is not large enough.", assert::light);
    }

    // Check that send displs and send counts are large enough
    KASSERT(
        *(send_counts.data() + this->size() - 1) +       // Last element of send_counts
                *(send_displs.data() + this->size() - 1) // Last element of send_displs
            <= asserting_cast<int>(send_buf.size()),
        assert::light
    );

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

    auto compute_required_recv_buf_size = [&]() {
        return compute_required_recv_buf_size_in_vectorized_communication(recv_counts, recv_displs, this->size());
    };

    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // Do the actual alltoallv
    [[maybe_unused]] int err = MPI_Alltoallv(
        send_buf.data(),    // sendbuf
        send_counts.data(), // sendcounts
        send_displs.data(), // sdispls
        mpi_send_type,      // sendtype
        recv_buf.data(),    // sendcounts
        recv_counts.data(), // recvcounts
        recv_displs.data(), // rdispls
        mpi_recv_type,      // recvtype
        mpi_communicator()  // comm
    );

    THROW_IF_MPI_ERROR(err, MPI_Alltoallv);

    return make_mpi_result(
        std::move(recv_buf),    // recv_buf
        std::move(recv_counts), // recv_counts
        std::move(recv_displs), // recv_displs
        std::move(send_displs)  // send_displs
    );
}
