// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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
#include "kamping/collectives/collectives_helpers.hpp"
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

/// @addtogroup kamping_collectives
/// @{

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
/// The following parameters are optional:
///  - \ref kamping::recv_buf() containing a buffer for the output. A buffer size of at least `send_recv_count` elements
///  is required.
///
///  - \ref kamping::send_recv_count() containing the number of elements to be processed in this operation. This
/// parameter has to be the same at each rank. If omitted, the size of the send buffer will be used as send_recv_count.
///
///  - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as data type in this operation. If omitted,
/// the \c MPI datatype is derived automatically based on send_buf's underlying \c value_type. If the type is provided
/// explicitly, the compatibility of the type and operation has to be ensured by the user.
///
///  - \ref kamping::values_on_rank_0() containing the value(s) that is/are returned in the \c recv_buf of rank 0. \c
///  values_on_rank_0 must be a container of the same size as \c recv_buf or a single value (which will be reused for
///  all elements of the \c recv_buf).
///
///
/// In-place exscan is supported by providing `send_recv_buf()` instead of `send_buf()` and `recv_buf()`. For details
/// on the in-place version, see \ref Communicator::exscan_inplace().
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional parameters described above.
/// @return Result object wrapping the output parameters to be returned by value.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>
/// \include{doc} docs/resize_policy.dox
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::exscan(Args... args) const {
    using namespace kamping::internal;
    constexpr bool inplace = internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    if constexpr (inplace) {
        return exscan_inplace(std::forward<Args>(args)...);
    } else {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, op),
            KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_recv_count, send_recv_type, values_on_rank_0)
        );

        // Get the send buffer and deduce the send and recv value types.
        auto const send_buf   = select_parameter_type<ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        KASSERT(
            is_same_on_all_ranks(send_buf.size()),
            "The send buffer has to be the same size on all ranks.",
            assert::light_communication
        );

        // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
        using default_recv_value_type = std::remove_const_t<send_value_type>;
        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto recv_buf =
            select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...)
                .template construct_buffer_or_rebind<DefaultContainerType>();

        // Get the send_recv_type.
        auto send_recv_type = determine_mpi_send_recv_datatype<send_value_type, decltype(recv_buf)>(args...);
        [[maybe_unused]] constexpr bool send_recv_type_is_in_param = !has_to_be_computed<decltype(send_recv_type)>;

        // Get the send_recv count
        using default_send_recv_count_type = decltype(kamping::send_recv_count_out());
        auto send_recv_count               = internal::select_parameter_type_or_default<
                                   internal::ParameterType::send_recv_count,
                                   default_send_recv_count_type>(std::tuple(), args...)
                                   .construct_buffer_or_rebind();

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

        // Resize the recv buffer to the same size as the send buffer; get the pointer needed for the MPI call.
        auto compute_required_recv_buf_size = [&]() {
            return asserting_cast<size_t>(send_recv_count.get_single_element());
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            // if the send_recv type is user provided, kamping cannot make any assumptions about the required size of
            // the recv buffer
            send_recv_type_is_in_param || recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        // Perform the MPI_Allreduce call and return.
        [[maybe_unused]] int err = MPI_Exscan(
            send_buf.data(),                      // sendbuf
            recv_buf.data(),                      // recvbuf,
            send_recv_count.get_single_element(), // count
            send_recv_type.get_single_element(),  // datatype,
            operation.op(),                       // op
            mpi_communicator()                    // communicator
        );
        this->mpi_error_hook(err, "MPI_Exscan");

        // MPI_Exscan leaves the recv_buf on rank 0 in an undefined state. We set it to the value provided via
        //  values_on_rank_0() if given. If values_on_rank_0() is not given and the operation is a built-in operation on
        //  a
        // built-in data-type, we set the value on rank 0 to the identity of that operation on that datatype (e.g. 0 for
        // addition on integers).
        if (rank() == 0) {
            constexpr bool has_values_on_rank_0_param = has_parameter_type<ParameterType::values_on_rank_0, Args...>();
            // We decided not to enforce having to provide values_on_rank_0() for a operation for which we cannot
            // auto-deduce the identity, as this would introduce a parameter which is required in some situtations in
            // KaMPIng, but never in MPI.
            if constexpr (has_values_on_rank_0_param) {
                auto values_on_rank_0_param =
                    select_parameter_type<ParameterType::values_on_rank_0>(args...).construct_buffer_or_rebind();
                KASSERT(
                    // if the send_recv type is user provided, kamping cannot make any assumptions about the required
                    // size of the recv buffer
                    (send_recv_type_is_in_param || values_on_rank_0_param.size() == 1
                     || values_on_rank_0_param.size() == asserting_cast<size_t>(send_recv_count.get_single_element())),
                    "on_rank_0 has to either be of size 1 or of the same size as the recv_buf.",
                    assert::light
                );
                if (values_on_rank_0_param.size() == 1) {
                    std::fill_n(
                        recv_buf.data(),
                        asserting_cast<size_t>(send_recv_count.get_single_element()),
                        *values_on_rank_0_param.data()
                    );
                } else {
                    std::copy_n(values_on_rank_0_param.data(), values_on_rank_0_param.size(), recv_buf.data());
                }
            } else if constexpr (operation.is_builtin) {
                std::fill_n(
                    recv_buf.data(),
                    asserting_cast<size_t>(send_recv_count.get_single_element()),
                    operation.identity()
                );
            }
        }

        return make_mpi_result<std::tuple<Args...>>(
            std::move(recv_buf),
            std::move(send_recv_count),
            std::move(send_recv_type)
        );
    }
}

/// @brief Wrapper for the in-place version of \ref Communicator::exscan().
///
/// This variant must be called collectively by all ranks in the communicator. It is semantically equivalent to \ref
/// Communicator::exscan(), but the input buffer is used as the output buffer. This means that the input buffer is
/// overwritten with the result of the exscan.
///
/// The following parameters are required:
/// - \ref kamping::send_recv_buf() containing the data for which to perform the exclusive scan and will store the
/// result of the scan.
///
/// - \ref kamping::op() wrapping the operation to apply to the input. If \ref kamping::send_recv_type() is provided
/// explicitly, the compatibility of the type and operation has to be ensured by the user.
///
/// The following parameters are optional:
/// - \ref kamping::send_recv_count() containing the number of elements to be processed in this operation. This
/// parameter has to be the same at each rank. If omitted, the size of the send buffer will be used as send_recv_count.
///
/// - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as data type in this operation. If omitted,
/// the \c MPI datatype is derived automatically based on send_recv_buf's underlying \c value_type.
///
///  - \ref kamping::values_on_rank_0() containing the value(s) that is/are written to the output on rank 0. \c
///  values_on_rank_0 must be a container of the same size as \c send_recv_buf or a single value (which will be reused
///  for all elements of the \c recv_buf).
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional parameters described above.
/// @return Result object wrapping the output parameters to be returned by value.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>
/// \include{doc} docs/resize_policy.dox
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::exscan_inplace(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_recv_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_count, send_recv_type, values_on_rank_0)
    );

    // get the send recv buffer and deduce the send and recv value types.
    auto send_recv_buf = select_parameter_type<ParameterType::send_recv_buf>(args...).construct_buffer_or_rebind();
    using value_type   = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;

    // get the send_recv_type
    auto type = determine_mpi_send_recv_datatype<value_type, decltype(send_recv_buf)>(args...);
    [[maybe_unused]] constexpr bool type_is_in_param = !has_to_be_computed<decltype(type)>;

    // get the send_recv count
    using default_count_type = decltype(kamping::send_recv_count_out());
    auto count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_count, default_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    constexpr bool do_compute_count = internal::has_to_be_computed<decltype(count)>;
    if constexpr (do_compute_count) {
        count.underlying() = asserting_cast<int>(send_recv_buf.size());
    }

    KASSERT(
        is_same_on_all_ranks(count.get_single_element()),
        "The send_recv_count has to be the same on all ranks.",
        assert::light_communication
    );

    // get the operation used for the reduction. The signature of the provided function is checked while building.
    auto& operation_param = select_parameter_type<ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<value_type>();

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(count.get_single_element());
    };
    send_recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        // if the send_recv type is user provided, kamping cannot make any assumptions about the required size of
        // the buffer
        type_is_in_param || send_recv_buf.size() >= compute_required_recv_buf_size(),
        "Send/Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // Perform the MPI_Exscan call and return.
    [[maybe_unused]] int err = MPI_Exscan(
        MPI_IN_PLACE,               // sendbuf
        send_recv_buf.data(),       // recvbuf,
        count.get_single_element(), // count
        type.get_single_element(),  // datatype,
        operation.op(),             // op
        mpi_communicator()          // communicator
    );
    this->mpi_error_hook(err, "MPI_Exscan");

    // MPI_Exscan leaves the recv_buf on rank 0 in an undefined state. We set it to the value provided via
    //  values_on_rank_0() if given. If values_on_rank_0() is not given and the operation is a built-in operation on
    //  a
    // built-in data-type, we set the value on rank 0 to the identity of that operation on that datatype (e.g. 0 for
    // addition on integers).
    if (rank() == 0) {
        constexpr bool has_values_on_rank_0_param = has_parameter_type<ParameterType::values_on_rank_0, Args...>();
        // We decided not to enforce having to provide values_on_rank_0() for a operation for which we cannot
        // auto-deduce the identity, as this would introduce a parameter which is required in some situtations in
        // KaMPIng, but never in MPI.
        if constexpr (has_values_on_rank_0_param) {
            auto values_on_rank_0_param =
                select_parameter_type<ParameterType::values_on_rank_0>(args...).construct_buffer_or_rebind();
            KASSERT(
                // if the send_recv type is user provided, kamping cannot make any assumptions about the required
                // size of the recv buffer
                (type_is_in_param || values_on_rank_0_param.size() == 1
                 || values_on_rank_0_param.size() == asserting_cast<size_t>(count.get_single_element())),
                "on_rank_0 has to either be of size 1 or of the same size as the recv_buf.",
                assert::light
            );
            if (values_on_rank_0_param.size() == 1) {
                std::fill_n(
                    send_recv_buf.data(),
                    asserting_cast<size_t>(count.get_single_element()),
                    *values_on_rank_0_param.data()
                );
            } else {
                std::copy_n(values_on_rank_0_param.data(), values_on_rank_0_param.size(), send_recv_buf.data());
            }
        } else if constexpr (operation.is_builtin) {
            std::fill_n(send_recv_buf.data(), asserting_cast<size_t>(count.get_single_element()), operation.identity());
        }
    }

    return make_mpi_result<std::tuple<Args...>>(std::move(send_recv_buf), std::move(count), std::move(type));
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
///  - \ref kamping::send_buf() containing the data for which to perform the exclusive scan. This buffer has to be a
///  single element on each rank.
///  - \ref kamping::op() the operation to apply to the input.
///
///  The following parameters are optional:
///  - \ref kamping::values_on_rank_0() containing the single value that is returned in the \c recv_buf of rank 0.///
///
///  @tparam Args Automatically deduced template parameters.
///  @param args All required and any number of the optional buffers described above.
///  @return The single element result of the exclusive scan.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
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

    using send_buf_type = buffer_type_with_requested_parameter_type<ParameterType::send_buf, Args...>;
    static_assert(
        send_buf_type::is_single_element,
        "The underlying container has to be a single element \"container\""
    );
    using value_type = typename send_buf_type::value_type;
    return this->exscan(recv_buf(alloc_new<value_type>), std::forward<Args>(args)...);
}
/// @}
