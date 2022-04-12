// This file is part of KaMPI.ng.
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

#include "kamping/checking_casts.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping::internal {

/// @brief CRTP mixin class for \c MPI_Gather.
///
/// This class is only to be used as a super class of kamping::Communicator
template <typename Communicator>
class Gather : public CRTPHelper<Communicator, Gather> {
public:
    /// @brief Wrapper for \c MPI_Gather
    ///
    /// This wrapper for \c MPI_Gather sends the same amount of data from each rank to a root. The following buffers are
    /// required:
    /// - \ref kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at
    /// each rank. See TODO gather_v if the amounts differ. The following buffers are optional:
    /// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c Communicator
    /// is used, see root().
    /// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, at the root, this buffer will contain
    /// all data from all send buffers. At all other ranks, the buffer will have size 0.
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @return Result type wrapping the output buffer if not specified as input parameter.
    template <typename... Args>
    auto gather(Args&&... args) {
        KAMPING_CHECK_PARAMETERS(
            Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(recv_buf, root));

        auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
        auto  send_buf        = send_buf_param.get();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;
        KASSERT(
            check_equal_sizes(send_buf.size()),
            "All PEs have to send the same number of elements. Use gatherv, if you want to send a different number of "
            "elements.",
            assert::light_communication);

        using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));

        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(), args...);
        using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

        auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
            std::tuple(this->underlying().root()), args...);
        KASSERT(this->underlying().is_valid_rank(root.rank()), "Invalid rank as root.");

        auto mpi_send_type = mpi_datatype<send_value_type>();
        auto mpi_recv_type = mpi_datatype<recv_value_type>();
        KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

        size_t recv_size     = (this->underlying().rank() == root.rank()) ? send_buf.size() : 0;
        size_t recv_buf_size = this->underlying().size() * recv_size;

        // error code can be unused if KTHROW is removed at compile time
        recv_buf.resize(recv_buf_size);
        [[maybe_unused]] int err = MPI_Gather(
            send_buf.data(), asserting_cast<int>(send_buf.size()), mpi_send_type, recv_buf.data(),
            asserting_cast<int>(recv_size), mpi_recv_type, root.rank_signed(), this->underlying().mpi_communicator());
        THROW_IF_MPI_ERROR(err, MPI_Gather);
        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{});
    }

protected:
    Gather() {}

private:
    bool check_equal_sizes(size_t local_size) const {
        std::vector<size_t> result(this->underlying().size(), 0);
        MPI_Gather(
            &local_size, 1, mpi_datatype<size_t>(), result.data(), 1, mpi_datatype<size_t>(),
            this->underlying().root_signed(), this->underlying().mpi_communicator());
        return std::equal(result.begin() + 1, result.end(), result.begin());
    }
}; // class Gather
} // namespace kamping::internal
