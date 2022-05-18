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

#include <numeric>
#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"

/// @brief Wrapper for \c MPI_Alltoall.
///
/// This wrapper for \c MPI_Alltoall sends the same amount of data from each rank to each rank. The following
/// buffers are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank and divisible by the size of the communicator. Each rank receives the same number of elements from
/// this buffer. Rank 0 receives the first `<buffer size>/<communicator size>` elements, rank 1 the next, and so
/// on. See alltoallv() if the amounts differ.
///
/// The following buffers are optional:
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
/// the data received as specified for send_buf. The data received from rank 0 comes first, followed by the data
/// received from rank 1, and so on.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::alltoall(Args&&... args) const {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(recv_buf));

    auto const& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    MPI_Datatype mpi_send_type    = mpi_datatype<send_value_type>();

    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<default_recv_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match.");
    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");
    KASSERT(mpi_send_type == mpi_recv_type, "The MPI receive type does not match the MPI send type.", assert::light);

    // Get the send and receive counts
    KASSERT(
        send_buf.size() % size() == 0lu,
        "The number of elements in send_buf is not divisible by the number of ranks in the communicator. Did you "
        "mean to use alltoallv?",
        assert::light);
    int send_count = asserting_cast<int>(send_buf.size() / size());

    size_t recv_buf_size = send_buf.size();
    int    recv_count    = asserting_cast<int>(recv_buf_size / size());
    KASSERT(send_count == recv_count, assert::light);
    recv_buf.resize(recv_buf_size);
    KASSERT(recv_buf_size == recv_buf.size(), assert::light);

    // These KASSERTs are required to avoid a false warning from g++ in release mode
    KASSERT(send_buf.data() != nullptr, assert::light);
    KASSERT(recv_buf.data() != nullptr, assert::light);

    [[maybe_unused]] int err = MPI_Alltoall(
        send_buf.data(), send_count, mpi_send_type, recv_buf.data(), recv_count, mpi_recv_type, mpi_communicator());

    THROW_IF_MPI_ERROR(err, MPI_Alltoall);
    return MPIResult(
        std::move(recv_buf),                // recv_buf
        internal::BufferCategoryNotUsed{},  // recv_counts
        internal::BufferCategoryNotUsed{},  // recv_count
        internal::BufferCategoryNotUsed{},  // recv_displs
        internal::BufferCategoryNotUsed{}); // send_displs
}

/// @brief Wrapper for \c MPI_Alltoallv.
///
/// This wrapper for \c MPI_Alltoallv sends the different amounts of data from each rank to each rank. The following
/// buffers are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at least
/// the sum of the send_counts argument.
/// - \ref kamping::send_counts() containing the number of elements to send to each rank.
///
/// The following parameters are optional but result in communication overhead if omitted:
/// - \ref kamping::recv_counts() containing the number of elements to receive from each rank.
///
/// The following buffers are optional:
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
/// the data received as specified for send_buf. The data received from rank 0 comes first, followed by the data
/// received from rank 1, and so on.
/// -\ref kamping::send_displs() containing the offsets of the messages in send_buf. The `send_counts[i]` elements
/// starting at `send_buf[send_displs[i]]` will be sent to rank `i`. If ommited, this is calculated as the exclusive
/// prefix-sum of `send_counts`.
/// -\ref kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]` elements
/// starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If ommited, this is calculated as the
/// exclusive prefix-sum of `recv_counts`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer, counts and displacements if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::alltoallv(Args&&... args) const {
    // Get all parameter objects
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
        KAMPING_OPTIONAL_PARAMETERS(recv_counts, recv_buf, send_displs, recv_displs));

    // Get send_buf
    auto const& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    MPI_Datatype mpi_send_type    = mpi_datatype<send_value_type>();

    // Get send_counts
    auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...);
    using send_counts_type  = typename std::remove_reference_t<decltype(send_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_counts_type>, int>, "Send counts must be of type int");
    KASSERT(send_counts.get().size() == this->size(), assert::light);

    // Get recv_counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(NewContainer<std::vector<int>>{}));
    auto&& recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(), args...);
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");

    // Get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<default_recv_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    // Get send_displs
    using default_send_displs_type = decltype(kamping::send_displs_out(NewContainer<std::vector<int>>{}));
    auto&& send_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
            std::tuple(), args...);
    using send_displs_type = typename std::remove_reference_t<decltype(send_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_displs_type>, int>, "Send displs must be of type int");

    // Get recv_displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(NewContainer<std::vector<int>>{}));
    auto&& recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(), args...);
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    // Check that send and receive buffers have matching types
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match.");
    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");
    KASSERT(mpi_send_type == mpi_recv_type, "The MPI receive type does not match the MPI send type.", assert::light);

    // Calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    if constexpr (do_calculate_recv_counts) {
        /// @todo make it possible to test whether this additional communication is skipped
        recv_counts.resize(this->size());
        auto recv_counts_span = recv_counts.get();
        auto send_counts_span = send_counts.get();
        this->alltoall(kamping::send_buf(send_counts_span), kamping::recv_buf(recv_counts_span));
    }
    KASSERT(recv_counts.get().size() == this->size(), assert::light);

    // Calculate send_displs if necessary
    constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;
    if constexpr (do_calculate_send_displs) {
        send_displs.resize(this->size());
        std::exclusive_scan(
            send_counts.get().data(), send_counts.get().data() + send_counts.get().size(), send_displs.get().data(), 0);
    }
    KASSERT(send_displs.get().size() == this->size(), assert::light);
    // Check that send displs and send counts match the size of send_buf
    KASSERT(
        *(send_counts.get().data() + send_counts.get().size() - 1) +       // Last element of send_counts
                *(send_displs.get().data() + send_displs.get().size() - 1) // Last element of send_displs
            <= asserting_cast<int>(send_buf.get().size()),
        assert::light);

    // Calculate recv_displs if necessary
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;
    if constexpr (do_calculate_recv_displs) {
        recv_displs.resize(this->size());
        std::exclusive_scan(
            recv_counts.get().data(), recv_counts.get().data() + recv_counts.get().size(), recv_displs.get().data(), 0);
    }
    KASSERT(recv_displs.get().size() == this->size(), assert::light);

    // Resize recv_buff
    int recv_buf_size = *(recv_counts.get().data() + recv_counts.get().size() - 1) + // Last element of recv_counts
                        *(recv_displs.get().data() + recv_displs.get().size() - 1);  // Last element of recv_displs
    recv_buf.resize(asserting_cast<size_t>(recv_buf_size));

    // Do the actual alltoallv
    [[maybe_unused]] int err = MPI_Alltoallv(
        send_buf.get().data(),    // sendbuf
        send_counts.get().data(), // sendcounts
        send_displs.get().data(), // sdispls
        mpi_send_type,            // sendtype
        recv_buf.get().data(),    // sendcounts
        recv_counts.get().data(), // recvcounts
        recv_displs.get().data(), // rdispls
        mpi_recv_type,            // recvtype
        mpi_communicator()        // comm
    );

    THROW_IF_MPI_ERROR(err, MPI_Alltoallv);

    return MPIResult(
        std::move(recv_buf),               // recv_buf
        std::move(recv_counts),            // recv_counts
        internal::BufferCategoryNotUsed{}, // recv_count
        std::move(recv_displs),            // recv_displs
        std::move(send_displs));           // send_displs
}
