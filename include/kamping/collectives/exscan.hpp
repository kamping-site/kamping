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

/// @brief Wrapper for \c MPI_Exscan.
///
/// \c exscan() wraps \c MPI_Exscan, which is used to perform an exclusive prefix reduction on data distributed across
/// the calling processes. \c exscan() returns in the \c recv_buf of the process with rank \f$i > 0\f$, the
/// reduction (calculated according to the function \c op) of the values in the \c send_bufs of processes with ranks
/// \f$0, \ldots, i - 1\f$ (i.e. excluding i as opposed to \c scan()). The value of the \c recv_buf on rank 0 is set to
/// the value of \c values_on_rank_0 if provided. If \c values_on_rank_0 is not provided and \c op is a built-in
/// operation on the data-type used, the value on rank 0 is set to the identity of that operation. If the operation is
/// not built-in on the data-type used and no \c values_on_rank_0() is provided, the contents of \c recv_buf on rank
/// 0 are undefined.
///
/// The following parameters are required:
///  - \ref kamping::send_buf() containing the data for which to perform the exclusive scan. This buffer has to be the
///  same size at each rank.
///  - \ref kamping::op() the operation to apply to the input.
///
///  The following parameters are optional:
///  - \ref kamping::recv_buf() containing a buffer for the output.
///  - \ref kamping::values_on_rank_0() containing the value(s) that is/are returned in the \c recv_buf of rank 0. \c
///  values_on_rank_0 must be a container of the same size as \c recv_buf or a single value (which will be reused for
///  all elements of the \c recv_buf).
///
///  @tparam Args Automatically deducted template parameters.
///  @param args All required and any number of the optional buffers described above.
///  @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::exscan(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, values_on_rank_0)
    );

    // Get the send buffer and deduce the send and recv value types.
    auto const& send_buf  = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    KASSERT(
        is_same_on_all_ranks(send_buf.size()),
        "The send buffer has to be the same size on all ranks.",
        assert::light_communication
    );

    // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match."
    );

    // Get the operation used for the reduction. The signature of the provided function is checked while building.
    auto& operation_param = select_parameter_type<ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<send_value_type>();

    // Resize the recv buffer to the same size as the send buffer; get the pointer needed for the MPI call.
    recv_buf.resize(send_buf.size());
    recv_value_type* recv_buf_ptr = recv_buf.data();
    KASSERT(recv_buf_ptr != nullptr, assert::light);
    KASSERT(recv_buf.size() == send_buf.size(), assert::light);
    // send_buf.size() is equal on all ranks, as checked above.

    // Perform the MPI_Allreduce call and return.
    [[maybe_unused]] int err = MPI_Exscan(
        send_buf.data(),                      // sendbuf
        recv_buf_ptr,                         // recvbuf,
        asserting_cast<int>(send_buf.size()), // count
        mpi_datatype<send_value_type>(),      // datatype,
        operation.op(),                       // op
        mpi_communicator()                    // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Reduce);

    // MPI_Exscan leaves the recv_buf on rank 0 in an undefined state. We set it to the value provided via
    //  values_on_rank_0() if given. If values_on_rank_0() is not given and the operation is a built-in operation on a
    // built-in data-type, we set the value on rank 0 to the identity of that operation on that datatype (e.g. 0 for
    // addition on integers).
    if (rank() == 0) {
        constexpr bool has_values_on_rank_0_param = has_parameter_type<ParameterType::values_on_rank_0, Args...>();
        // We decided not to enforce having to provide values_on_rank_0() for a operation for which we cannot
        // auto-deduce the identity, as this would introduce a parameter which is required in some situtations in
        // KaMPIng, but never in MPI.
        if constexpr (has_values_on_rank_0_param) {
            auto const& values_on_rank_0_param = select_parameter_type<ParameterType::values_on_rank_0>(args...);
            KASSERT(
                (values_on_rank_0_param.size() == 1 || values_on_rank_0_param.size() == recv_buf.size()),
                "on_rank_0 has to either be of size 1 or of the same size as the recv_buf.",
                assert::light
            );
            if (values_on_rank_0_param.size() == 1) {
                std::fill_n(recv_buf.data(), recv_buf.size(), *values_on_rank_0_param.data());
            } else {
                std::copy_n(values_on_rank_0_param.data(), values_on_rank_0_param.size(), recv_buf.data());
            }
        } else if constexpr (operation.is_builtin) {
            std::fill_n(recv_buf.data(), recv_buf.size(), operation.identity());
        }
    }

    return make_mpi_result(std::move(recv_buf));
}

/// @brief Wrapper for \c MPI_exscan for single elements.
///
/// This is functionally equivalent to \c exscan() but provided for uniformity with other operations (e.g. \c
/// bcast_single()). \c exscan_single() wraps \c MPI_Exscan, which is used to perform an exclusive prefix reduction on
/// data distributed across the calling processes. \c exscan_single() returns on the process with
/// rank \f$i > 0\f$, the reduction (calculated according to the function \c op) of the values in the \c send_bufs of
/// processes with ranks \f$0, \ldots, i - 1\f$ (i.e. excluding i as opposed to \c scan()). The result
/// on rank 0 is set to the value of \c values_on_rank_0 if provided. If \c values_on_rank_0 is not provided and \c op
/// is a built-in operation on the data-type used, the value on rank 0 is set to the identity of that operation. If the
/// operation is not built-in on the data-type used and no \c values_on_rank_0() is provided, the result on rank 0 is
/// undefined.
///
/// The following parameters are required:
///  - \ref kamping::send_buf() containing the data for which to perform the exclusive scan. This buffer has to be of
///  size 1 on each rank.
///  - \ref kamping::op() the operation to apply to the input.
///
///  The following parameters are optional:
///  - \ref kamping::values_on_rank_0() containing the single value that is returned in the \c recv_buf of rank 0.///
///
///  @tparam Args Automatically deducted template parameters.
///  @param args All required and any number of the optional buffers described above.
///  @return The single element result of the exclusive scan.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::exscan_single(Args... args) const {
    //! If you expand this function to not being only a simple wrapper around exscan, you have to write more unit
    //! tests!

    using namespace kamping::internal;

    // The send and recv buffers are always of the same size in exscan, thus, there is no additional exchange of
    // recv_counts.
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(values_on_rank_0)
    );

    KASSERT(
        select_parameter_type<ParameterType::send_buf>(args...).size() == 1u,
        "The send buffer has to be of size 1 on all ranks.",
        assert::light
    );

    return this->exscan(std::forward<Args>(args)...).extract_recv_buffer()[0];
}
