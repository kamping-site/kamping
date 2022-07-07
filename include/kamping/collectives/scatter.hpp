// This file is part of KaMPIng
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

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace {
// Broadcasts a value from one PE to all PEs.
template <typename T>
int bcast_value(kamping::Communicator const& comm, T const bcast_value, int const root) {
    using namespace kamping::internal;
    using namespace kamping;
    T                          bcast_result = bcast_value;
    [[maybe_unused]] int const result = MPI_Bcast(&bcast_result, 1, mpi_datatype<T>(), root, comm.mpi_communicator());
    THROW_IF_MPI_ERROR(result, MPI_Bcast);
    return bcast_result;
}
} // anonymous namespace

/// @brief Wrapper for \c MPI_Scatter.
///
/// This wrapper for \c MPI_Scatter distributes data on the root PE evenly across all PEs in the current
/// communicator.
///
/// The following parameters are mandatory:
/// - \ref kamping::send_buf() containing the data to be evenly distributed across all PEs. The size of
/// this buffer must be divisible by the number of PEs in the current communicator. Non-root PEs can omit a send
/// buffer by passing `kamping::ignore` to \ref kamping::send_buf().
///
/// The following parameters are optional but incur communication overhead if omitted:
/// - \ref kamping::recv_count() specifying the number of elements sent to each PE. If this parameter is omitted,
/// the number of elements sent to each PE is computed based on the size of the \ref kamping::send_buf() on the root
/// PE and broadcasted to other PEs.
///
/// The following parameters are optional:
/// - \ref kamping::root() specifying the rank of the root PE. If omitted, the default root PE of the communicator
/// is used instead.
/// - \ref kamping::recv_buf() containing the received data. If omitted, a new buffer is allocated and returned.
///
/// @tparam Args Deduced template parameters.
/// @param args Required and optionally optional parameters.
/// @return kamping::MPIResult wrapping the output buffer if not specified as an input parameter.
template <typename... Args>
auto kamping::Communicator::scatter(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(root, recv_buf, recv_count));

    // Optional parameter: root()
    // Default: communicator root
    using root_param_type = decltype(kamping::root(0));
    auto&& root_param     = internal::select_parameter_type_or_default<internal::ParameterType::root, root_param_type>(
        std::tuple(root()), args...);
    size_t const root     = root_param.rank();
    int const    int_root = root_param.rank_signed();
    KASSERT(is_valid_rank(root), "Invalid root rank " << root << " in communicator of size " << size(), assert::light);
    KASSERT(this->is_same_on_all_ranks(root), "Root has to be the same on all ranks.", assert::light_communication);

    // Mandatory parameter send_buf()
    auto send_buf              = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
    using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
    auto const*  send_buf_ptr  = send_buf.data();
    KASSERT((!is_root(root) || send_buf_ptr != nullptr), "Send buffer must be specified on root.", assert::light);

    // Compute sendcount based on the size of the sendbuf
    KASSERT(
        send_buf.size() % size() == 0u, "Size of the send buffer ("
                                            << send_buf.size() << ") is not divisible by the number of PEs (" << size()
                                            << ") in the communicator.");
    int const send_count = asserting_cast<int>(send_buf.size() / size());

    // Optional parameter: recv_buf()
    // Default: allocate new container
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    // Make sure that send and recv buffers use the same type
    static_assert(
        std::is_same_v<send_value_type, recv_value_type>, "Mismatching send_buf() and recv_buf() value types.");

    // Optional parameter: recv_count()
    // Default: compute value based on send_buf.size on root

    auto&& recv_count_param = internal::select_parameter_type_or_default<
        internal::ParameterType::recv_count,
        LibAllocatedSingleElementBuffer<int, internal::ParameterType::recv_count, internal::BufferType::in_buffer>>(
        std::tuple(), args...);

    constexpr bool is_output_parameter = has_to_be_computed<decltype(recv_count_param)>;

    KASSERT(
        is_same_on_all_ranks(is_output_parameter),
        "recv_count() parameter is an output parameter on some PEs, but not on alle PEs.", assert::light_communication);

    // If it is an output parameter, broadcast send_count to get recv_count
    if constexpr (is_output_parameter) {
        *recv_count_param.get().data() = bcast_value(*this, send_count, int_root);
    }

    int recv_count = *recv_count_param.get().data();

    // Validate against send_count
    KASSERT(
        recv_count == bcast_value(*this, send_count, int_root), "Specified recv_count() does not match the send count.",
        assert::light_communication);

    recv_buf.resize(static_cast<std::size_t>(recv_count));
    auto* recv_buf_ptr = recv_buf.data();

    [[maybe_unused]] int const err = MPI_Scatter(
        send_buf_ptr, send_count, mpi_send_type, recv_buf_ptr, recv_count, mpi_recv_type, int_root, mpi_communicator());
    THROW_IF_MPI_ERROR(err, MPI_Scatter);

    return MPIResult(
        std::move(recv_buf), internal::BufferCategoryNotUsed{}, std::move(recv_count_param),
        internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{});
}
