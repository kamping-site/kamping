
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

/// @brief Wrapper for \c MPI_Scan.
///
/// This wraps \c MPI_Scan, which is used to perform an inclusive prefix reduction on data distributed across the
/// calling processes. / \c scan() returns in \c recv_buf of the process with rank \c i, the reduction (calculated
/// according to the function op) of the values in the sendbufs of processes with ranks \f$0, ..., i\f$ (inclusive).
///
/// The following parameters are required:
/// - kamping::send_buf() containing the data for which to perform the exclusive scan. This buffer has to be the
///  same size at each rank.
/// - kamping::op() wrapping the operation to apply to the input.
///
/// The following parameters are optional:
/// - kamping::recv_buf() containing a buffer for the output. The buffer will be resized according to the buffer's
/// kamping::BufferResizePolicy. If this is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to hold all received elements. This requires a size of at least `send_recv_count`.
///
/// @todo replace with send_recv_count once this has been implemented by niklas
/// - kamping::send_counts() containing the number of elements to send. This parameter has to be an integer and has to
/// be the same at each rank.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::scan(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(send_counts, recv_buf)
    );

    // Get the send buffer and deduce the send and recv value types.
    auto const& send_buf          = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match."
    );

    // Get the send_recv count
    using default_send_recv_count_type = decltype(kamping::send_counts_out(alloc_new<int>));
    auto&& send_recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_recv_count_type>(
            std::tuple(),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_recv_count)>::is_single_element,
        "send_recv_count() parameter must be a single value."
    );
    constexpr bool do_compute_send_recv_count = internal::has_to_be_computed<decltype(send_recv_count)>;
    if constexpr (do_compute_send_recv_count) {
        send_recv_count.underlying() = asserting_cast<int>(send_buf.size());
    }

    KASSERT(
        is_same_on_all_ranks(send_recv_count.get_single_element()),
        "The send_recv_count has to be the same on all ranks.",
        assert::light_communication
    );

    // Get the operation used for the reduction. The signature of the provided function is checked while building.
    auto& operation_param = select_parameter_type<ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<send_value_type>();

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(send_recv_count.get_single_element());
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // Perform the MPI_Scan call and return.
    [[maybe_unused]] int err = MPI_Scan(
        send_buf.data(),                      // sendbuf
        recv_buf.data(),                      // recvbuf,
        send_recv_count.get_single_element(), // count
        mpi_datatype<send_value_type>(),      // datatype,
        operation.op(),                       // op
        mpi_communicator()                    // communicator
    );

    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return make_mpi_result(std::move(recv_buf));
}

/// @brief Wrapper for \c MPI_Scan for single elements.
///
/// This is functionally equivalent to \c scan() but provided for uniformity with other operations (e.g. \c
/// bcast_single()). \c scan_single() wraps \c MPI_Scan, which is used to perform an inclusive prefix reduction on data
/// distributed across the calling processes. \c scan() returns on the process with rank \f$i\f$, the
/// reduction (calculated according to the function op) of the values in the sendbufs of processes with ranks \f$0, ...,
/// i\f$ (inclusive).
///
/// The following parameters are required:
/// - kamping::send_buf() containing the data for which to perform the exclusive scan. This buffer has to be of
/// size 1 on each rank.
/// - kamping::op() wrapping the operation to apply to the input.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return The single element result of the scan.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::scan_single(Args... args) const {
    //! If you expand this function to not being only a simple wrapper around scan, you have to write more unit
    //! tests!

    using namespace kamping::internal;

    // The send and recv buffers are always of the same size in scan, thus, there is no additional exchange of
    // recv_counts.
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS());

    KASSERT(
        select_parameter_type<ParameterType::send_buf>(args...).size() == 1u,
        "The send buffer has to be of size 1 on all ranks.",
        assert::light
    );
    using value_type =
        typename std::remove_reference_t<decltype(select_parameter_type<ParameterType::send_buf>(args...))>::value_type;
    return this->scan(recv_buf(alloc_new<value_type>), std::forward<Args>(args)...).extract_recv_buffer();
}
