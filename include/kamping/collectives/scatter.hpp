// This file is part of KaMPI.ng
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <mpi.h>
#include <type_traits>

#include "kamping/checking_casts.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"

namespace kamping::internal {
/// @brief CRTP mixin class for \c MPI_Scatter.
///
/// This class is only to be used as a super class of kamping::Communicator.
template <typename Communicator>
class Scatter : public CRTPHelper<Communicator, Scatter> {
public:
    /// @brief Wrapper for \c MPI_Scatter.
    ///
    /// This wrapper for \c MPI_Scatter distributes data on the root PE evenly across all PEs in the current
    /// communicator.
    ///
    /// The following parameters are mandatory:
    /// - (root only) \ref kamping::send_buf() containing the data to be evenly distributed across all PEs. The size of
    /// this buffer must be divisible by the number of PEs in the current communicator.
    ///
    /// The following parameters are optional:
    /// - \ref kamping::recv_count() specifying the number of elements sent to each PE. If this parameter is omitted,
    /// the number of elements sent to each PE is computed based on the size of the \ref kamping::send_buf() on the root
    /// PE and broadcasted to other PEs.
    /// - \ref kamping::root() specifying the rank of the root PE. If omitted, the default root PE of the communicator
    /// is used instead.
    /// - \ref kamping::recv_buf() containing the received data. If omitted, a new buffer is allocated and returned.
    ///
    /// @tparam Args Deduced template parameters.
    /// @param args Required and optionally optional parameters.
    /// @return Result type wrapping the output buffer if not specified as an input parameter.
    template <typename... Args>
    auto scatter(Args&&... args) {
        static_assert(
            all_parameters_are_rvalues<Args...>,
            "All parameters have to be passed in as rvalue references, meaning that you must not hold a variable "
            "returned by the named parameter helper functions like send_buf().");

        // Parameter send_buf(): required on root, optional/ignored otherwise
        auto send_buf              = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
        using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
        auto const*  send_buf_ptr  = send_buf.data();
        KASSERT(
            !this->comm().is_root() || send_buf_ptr != nullptr, "Send buffer must be specified on root.",
            assert::light);

        // Compute sendcount based on the size of the sendbuf
        KASSERT(
            send_buf.size % static_cast<std::size_t>(this->comm().size()) == 0u,
            "Size of the send buffer (" << send_buf.size << ") is not divisible by the number of PEs (" << comm().size()
                                        << ") in the communicator.");
        int const send_count = asserting_cast<int>(send_buf.size / static_cast<std::size_t>(comm().size()));

        // Optional parameter: root()
        // Default: communicator root
        using root_param_type = decltype(kamping::root(0));
        auto&& root_param = internal::select_parameter_type_or_default<internal::ParameterType::root, root_param_type>(
            std::tuple(comm().root()), args...);
        int const root = root_param.rank();
        KASSERT(
            comm().is_valid_rank(root), "Invalid root rank " << root << " in communicator of size " << comm().size(),
            assert::light);

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
            std::is_same_v<send_value_type, recv_value_type>, "Mismatching send and receive buffer value types");

        // Optional parameter: recv_count()
        // Default: compute value based on send_buf.size on root
        int recv_count = 0;

        if constexpr (internal::has_parameter_type<internal::ParameterType::recv_count, Args...>()) {
            recv_count = internal::select_parameter_type<internal::ParameterType::recv_count>(args...).recv_count();

            // Validate against send_count
            KASSERT(
                recv_count == bcast_recv_count(send_count, root), "Specified recv count does not match the send count.",
                assert::light_communication);
        } else {
            // Broadcast send_count to get recv_count
            recv_count = this->bcast_recv_count(send_count, root);
        }

        auto* recv_buf_ptr = recv_buf.get_ptr(static_cast<std::size_t>(recv_count));

        [[maybe_unused]] int const err = MPI_Scatter(
            send_buf_ptr, send_count, mpi_send_type, recv_buf_ptr, recv_count, mpi_recv_type, root,
            comm().mpi_communicator());
        THROW_IF_MPI_ERROR(err, MPI_Scatter);

        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{});
    }

protected:
    // Prevent class instantiation by making the ctor protected.
    Scatter() = default;

private:
    // Broadcast recv count from root PE to all PEs.
    int bcast_recv_count(int const bcast_value, int root) {
        int                        bcast_result = bcast_value;
        [[maybe_unused]] int const result       = MPI_Bcast(&bcast_result, 1, MPI_INT, root, comm().mpi_communicator());
        THROW_IF_MPI_ERROR(result, MPI_Bcast);
        return bcast_result;
    }

    // Return communicator.
    Communicator const& comm() const {
        return this->underlying();
    }
};
} // namespace kamping::internal
