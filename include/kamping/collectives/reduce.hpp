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

#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Reduce.
///
/// This wraps \c MPI_Reduce. The operation combines the elements in the input buffer provided via \c
/// kamping::send_buf() and returns the combined value on the root rank. The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank.
/// - \ref kamping::op() wrapping the operation to apply to the input.
///
/// The following parameters are optional:
/// - \ref kamping::send_counts() specifiying how many elements of the buffer take part in the reduction.
/// This parameter has to be an integer. If ommited, this size of the send buffer is used as a default.
/// - \ref kamping::recv_buf() containing a buffer for the output.
/// - \ref kamping::root() the root rank. If not set, the default root process of the communicator will be used.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::reduce(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_counts, root)
    );

    // Get all parameters
    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );

    auto const& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    auto& operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
    // If you want to understand the syntax of the following line, ignore the "template " ;-)
    auto operation = operation_param.template build_operation<send_value_type>();

    using default_send_count_type = decltype(kamping::send_counts_out(alloc_new<int>));
    auto&& send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_count_type>(
            {},
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_count)>::is_single_element,
        "send_counts() parameter must be a single value."
    );
    if constexpr (has_to_be_computed<decltype(send_count)>) {
        send_count.underlying() = asserting_cast<int>(send_buf.size());
    }

    // Check parameters
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match."
    );
    MPI_Datatype type = mpi_datatype<send_value_type>();

    KASSERT(is_valid_rank(root.rank_signed()), "The provided root rank is invalid.", assert::light);
    KASSERT(
        this->is_same_on_all_ranks(root.rank_signed()),
        "Root has to be the same on all ranks.",
        assert::light_communication
    );
    auto compute_required_recv_buf_size = [&] {
        return asserting_cast<size_t>(send_count.get_single_element());
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );
    // from the standard:
    // > The routine is called by all group members using the same arguments for count, datatype, op,
    // > root and comm.
    KASSERT(
        this->is_same_on_all_ranks(send_count.get_single_element()),
        "send_count() has to be the same on all ranks.",
        assert::light_communication
    );

    [[maybe_unused]] int err = MPI_Reduce(
        send_buf.data(),                 // send_buf
        recv_buf.data(),                 // recv_buf
        send_count.get_single_element(), // count
        type,                            // type
        operation.op(),                  // op
        root.rank_signed(),              // root
        mpi_communicator()               // comm
    );

    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return make_mpi_result(std::move(recv_buf), std::move(send_count));
}
