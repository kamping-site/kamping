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

#include <tuple>
#include <type_traits>

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

/// @brief Wrapper for \c MPI_Allreduce; which is semantically a reduction followed by a broadcast.
///
/// This wraps \c MPI_Allreduce. The operation combines the elements in the input buffer provided via \c
/// kamping::send_buf() and returns the combined value on all ranks.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to have the same size at
/// each rank.
/// - \ref kamping::op() wrapping the operation to apply to the input. If \ref kamping::send_recv_type() is provided
/// explicitly, the compatibility of the type and operation has to be ensured by the user.
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() containing a buffer for the output.
///
/// - \ref kamping::send_recv_count() specifying how many elements of the send buffer take part in the reduction. If
/// omitted, the size of send buffer is used. This parameter is mandatory if \ref kamping::send_recv_type() is given.
///
/// - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype
/// is derived automatically based on send_buf's underlying \c value_type.
///
/// In-place allreduce is supported by providing `send_recv_buf()` instead of `send_buf()` and `recv_buf()`. For details
/// on the in-place version, see \ref Communicator::allreduce_inplace().
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::allreduce(Args... args) const {
    using namespace kamping::internal;
    constexpr bool inplace = internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    if constexpr (inplace) {
        return this->allreduce_inplace(std::forward<Args>(args)...);
    } else {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, op),
            KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_recv_count, send_recv_type)
        );

        // Get the send buffer and deduce the send and recv value types.
        auto const send_buf   = select_parameter_type<ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        using default_recv_value_type = std::remove_const_t<send_value_type>;

        // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto recv_buf =
            select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...)
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
        static_assert(
            std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
            "Types of send and receive buffers do not match."
        );

        // Get the send_recv_type.
        auto send_recv_type = determine_mpi_send_recv_datatype<send_value_type, decltype(recv_buf)>(args...);
        [[maybe_unused]] constexpr bool send_recv_type_is_in_param = !has_to_be_computed<decltype(send_recv_type)>;

        // Get the operation used for the reduction. The signature of the provided function is checked while building.
        auto& operation_param = select_parameter_type<ParameterType::op>(args...);
        auto  operation       = operation_param.template build_operation<send_value_type>();

        using default_send_recv_count_type = decltype(kamping::send_recv_count_out());
        auto send_recv_count               = internal::select_parameter_type_or_default<
                                   internal::ParameterType::send_recv_count,
                                   default_send_recv_count_type>({}, args...)
                                   .construct_buffer_or_rebind();
        if constexpr (has_to_be_computed<decltype(send_recv_count)>) {
            send_recv_count.underlying() = asserting_cast<int>(send_buf.size());
        }
        // from the standard:
        // > The routine is called by all group members using the same arguments for count, datatype, op,
        // > root and comm.
        KASSERT(
            this->is_same_on_all_ranks(send_recv_count.get_single_element()),
            "send_count() has to be the same on all ranks.",
            assert::light_communication
        );

        auto compute_required_recv_buf_size = [&] {
            return asserting_cast<size_t>(send_recv_count.get_single_element());
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);

        KASSERT(
            // if the send type is user provided, kamping cannot make any assumptions about the required size of the
            // recv buffer
            send_recv_type_is_in_param || recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        // Perform the MPI_Allreduce call and return.
        [[maybe_unused]] int err = MPI_Allreduce(
            send_buf.data(),                      // sendbuf
            recv_buf.data(),                      // recvbuf,
            send_recv_count.get_single_element(), // count
            send_recv_type.get_single_element(),  // datatype,
            operation.op(),                       // op
            mpi_communicator()                    // communicator
        );

        this->mpi_error_hook(err, "MPI_Allreduce");
        return make_mpi_result<std::tuple<Args...>>(
            std::move(recv_buf),
            std::move(send_recv_count),
            std::move(send_recv_type)
        );
    }
}

/// @brief Wrapper for the in-place version of \ref Communicator::allreduce().
///
/// This variant must be called collectively by all ranks in the communicator. It is semantically equivalent to \ref
/// Communicator::allreduce(), but the input buffer is used as the output buffer. This means that the input buffer is
/// overwritten with the result of the reduction.
///
/// The following parameters are required:
/// - \ref kamping::send_recv_buf() containing the data that is sent to each rank and will store the result of the
/// reduction.
///
/// - \ref kamping::op() wrapping the operation to apply to the input. If \ref kamping::send_recv_type() is provided
/// explicitly, the compatibility of the type and operation has to be ensured by the user.
///
/// The following parameters are optional:
/// - \ref kamping::send_recv_count() specifying how many elements of the send buffer take part in the reduction. If
/// omitted, the size of `send_recv_buf` is used. This parameter is mandatory if \ref kamping::send_recv_type() is
/// given.
///
/// - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype
/// is derived automatically based on `send_recv_buf`'s underlying \c value_type.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::allreduce_inplace(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_recv_buf, op),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_count, send_recv_type)
    );

    // Get the send buffer and deduce the send and recv value types.
    auto send_recv_buf = select_parameter_type<ParameterType::send_recv_buf>(args...).construct_buffer_or_rebind();
    using send_recv_value_type = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;

    // Get the send_recv_type.
    auto send_recv_type = determine_mpi_send_recv_datatype<send_recv_value_type, decltype(send_recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_recv_type_is_in_param = !has_to_be_computed<decltype(send_recv_type)>;

    // Get the operation used for the reduction. The signature of the provided function is checked while building.
    auto& operation_param = select_parameter_type<ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<send_recv_value_type>();

    using default_send_recv_count_type = decltype(kamping::send_recv_count_out());
    auto send_recv_count               = internal::select_parameter_type_or_default<
                               internal::ParameterType::send_recv_count,
                               default_send_recv_count_type>({}, args...)
                               .construct_buffer_or_rebind();
    if constexpr (has_to_be_computed<decltype(send_recv_count)>) {
        send_recv_count.underlying() = asserting_cast<int>(send_recv_buf.size());
    }
    // from the standard:
    // > The routine is called by all group members using the same arguments for count, datatype, op,
    // > root and comm.
    KASSERT(
        this->is_same_on_all_ranks(send_recv_count.get_single_element()),
        "send_recv_count() has to be the same on all ranks.",
        assert::light_communication
    );

    KASSERT(
        // if the send type is user provided, kamping cannot make any assumptions about the required size of the
        // recv buffer
        send_recv_type_is_in_param
            || send_recv_buf.size() >= asserting_cast<size_t>(send_recv_count.get_single_element()),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // Perform the MPI_Allreduce call and return.
    [[maybe_unused]] int err = MPI_Allreduce(
        MPI_IN_PLACE,                         // sendbuf
        send_recv_buf.data(),                 // recvbuf
        send_recv_count.get_single_element(), // count
        send_recv_type.get_single_element(),  // datatype,
        operation.op(),                       // op
        mpi_communicator()                    // communicator
    );

    this->mpi_error_hook(err, "MPI_Allreduce");
    return make_mpi_result<std::tuple<Args...>>(
        std::move(send_recv_buf),
        std::move(send_recv_count),
        std::move(send_recv_type)
    );
}

/// @brief Wrapper for \c MPI_Allreduce; which is semantically a reduction followed by a broadcast.
///
/// Calling allreduce_single() is a shorthand for calling allreduce() with a \ref kamping::send_buf() of size 1. It
/// always issues only a single <code>MPI_Allreduce</code> call, as no receive counts have to be exchanged.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to wrap a single element
/// on each rank.
/// - \ref kamping::op() wrapping the operation to apply to the input.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional parameters described above.
/// @return The single output value.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allreduce_single(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS());

    using send_buf_type = buffer_type_with_requested_parameter_type<ParameterType::send_buf, Args...>;
    static_assert(
        send_buf_type::is_single_element,
        "The underlying container has to be a single element \"container\""
    );

    using value_type = typename std::remove_reference_t<
        decltype(select_parameter_type<ParameterType::send_buf>(args...).construct_buffer_or_rebind())>::value_type;
    return this->allreduce(recv_buf(alloc_new<value_type>), std::forward<Args>(args)...);
}
/// @}
