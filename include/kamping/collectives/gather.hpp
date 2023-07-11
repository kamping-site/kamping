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

#include <cstddef>
#include <numeric>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

/// @brief Wrapper for \c MPI_Gather.
///
/// This wrapper for \c MPI_Gather collects the same amount of data from each rank to a root.
///
/// The following arguments are required:
/// - \ref kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at
/// each rank. See gather_v if the amounts differ. The following buffers are optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c Communicator
/// is used, see root().
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, at the root, this buffer will contain
/// all data from all send buffers. At all other ranks, the buffer will have size 0.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::gather(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(recv_buf, root));

    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;
    KASSERT(
        is_same_on_all_ranks(send_buf.size()),
        "All PEs have to send the same number of elements. Use gatherv, if you want to send a different number of "
        "elements.",
        assert::light_communication
    );

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));

    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    KASSERT(this->is_valid_rank(root.rank_signed()), "Invalid rank as root.");
    KASSERT(
        this->is_same_on_all_ranks(root.rank_signed()),
        "Root has to be the same on all ranks.",
        assert::light_communication
    );

    auto mpi_send_type = mpi_datatype<send_value_type>();
    auto mpi_recv_type = mpi_datatype<recv_value_type>();
    KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

    size_t recv_size     = (this->rank_signed() == root.rank_signed()) ? send_buf.size() : 0;
    size_t recv_buf_size = this->size() * recv_size;

    // error code can be unused if KTHROW is removed at compile time
    recv_buf.resize(recv_buf_size);
    [[maybe_unused]] int err = MPI_Gather(
        send_buf.data(),                      // sendbuffer
        asserting_cast<int>(send_buf.size()), // sendcount
        mpi_send_type,                        // sendtype
        recv_buf.data(),                      // recvbuffer
        asserting_cast<int>(recv_size),       // recvcount
        mpi_recv_type,                        // recvtype
        root.rank_signed(),                   // root
        this->mpi_communicator()              // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Gather);
    return make_mpi_result(std::move(recv_buf));
}

/// @brief Wrapper for \c MPI_Gatherv.
///
/// This wrapper for \c MPI_Gatherv collects possibly different amounts of data from each rank to a root.
///
/// The following arguments are required:
/// - \ref kamping::send_buf() containing the data that is sent to the root.
///
/// The following parameter is optional but results in communication overhead if omitted:
/// - \ref kamping::recv_counts() containing the number of elements to receive from each rank. Only the root rank uses
/// the content of this buffer, all other ranks ignore it. However, if provided on any rank it must be provided on all
/// ranks (possibly empty on non-root ranks).
///
/// The following buffers are optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c Communicator
/// is used, see root().
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, at the root, this buffer will contain
/// all data from all send buffers. At all other ranks, the buffer will have size 0.
/// - \ref kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]` elements
/// starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If omitted, this is calculated as the
/// exclusive prefix-sum of `recv_counts`.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::gatherv(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, root, recv_counts, recv_displs)
    );

    // get send buffer
    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

    // get recv buffer
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    // get root rank
    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    KASSERT(this->is_valid_rank(root.rank_signed()), "Invalid rank as root.");
    KASSERT(
        this->is_same_on_all_ranks(root.rank_signed()),
        "Root has to be the same on all ranks.",
        assert::light_communication
    );

    // get recv counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        );
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");

    // get recv displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
    auto&& recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(),
            args...
        );
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    // calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_counts),
        "Receive counts are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_counts) {
        recv_counts.resize(this->size());
        this->gather(
            kamping::send_buf(static_cast<int>(send_buf.size())),
            kamping::recv_buf(recv_counts.get()),
            kamping::root(root.rank_signed())
        );
    }

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));

    // calculate recv_displs if necessary
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;
    if constexpr (do_calculate_recv_displs) {
        if (this->is_root(root.rank_signed())) {
            recv_displs.resize(this->size());
            std::exclusive_scan(recv_counts.data(), recv_counts.data() + recv_counts.size(), recv_displs.data(), 0);
        }
    }
    auto mpi_send_type = mpi_datatype<send_value_type>();
    auto mpi_recv_type = mpi_datatype<recv_value_type>();
    KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

    if (this->is_root(root.rank_signed())) {
        KASSERT(recv_counts.size() == this->size(), assert::light);
        KASSERT(recv_displs.size() == this->size(), assert::light);
        int const recv_buf_size = *(recv_counts.data() + recv_counts.size() - 1) + // Last element of recv_counts
                                  *(recv_displs.data() + recv_displs.size() - 1);  // Last element of recv_displs
        recv_buf.resize(asserting_cast<size_t>(recv_buf_size));
    } else {
        recv_buf.resize(0);
    }

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Gatherv(
        send_buf.data(),                      // send buffer
        asserting_cast<int>(send_buf.size()), // send count
        mpi_send_type,                        // send type
        recv_buf.data(),                      // recv buffer
        recv_counts.data(),                   // recv counts
        recv_displs.data(),                   // recv displacmenets
        mpi_recv_type,                        // recv type
        root.rank_signed(),                   // root rank
        this->mpi_communicator()              // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Gather);
    return make_mpi_result(std::move(recv_buf), std::move(recv_counts), std::move(recv_displs));
}
