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

#include <numeric>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Allgather.
///
/// This wrapper for \c MPI_Allgather collects the same amount of data from each rank to all ranks. It is semantically
/// equivalent to performing a \c gather() followed by a broadcast of the collected data.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at
/// each rank. See allgather_v if the amounts differ.
///
/// The following parameters are optional:
/// - \ref kamping::send_count() specifying how many elements are sent. If
/// omitted, the size of the send buffer is used. This parameter is mandatory if \ref kamping::send_type() is given.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::recv_count() specifying how many elements are received. If
/// omitted, the value of send_counts will be used. This parameter is mandatory if \ref kamping::recv_type() is given.
///
/// - \ref kamping::recv_buf() specifying a buffer for the output. Afterwards, this buffer will contain
/// all data from all send buffers. This requires a size of the buffer of at least `recv_counts * communicator size`.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c MPI datatype is
/// derived automatically based on recv_buf's underlying \c value_type.
///
/// In-place allgather is supported by passing send_recv_buf() as parameter. This changes the requirements for the other
/// parameters, see \ref Communicator::allgather_inplace.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgather(Args... args) const {
    using namespace kamping::internal;
    constexpr bool inplace = internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    if constexpr (inplace) {
        return allgather_inplace(std::forward<Args>(args)...);
    } else {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf),
            KAMPING_OPTIONAL_PARAMETERS(send_count, recv_count, recv_buf, send_type, recv_type)
        );

        // get the send/recv buffer and types
        auto send_buf_param =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        auto send_buf         = send_buf_param.get();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

        using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
        auto recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

        auto [send_type, recv_type] =
            internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
        [[maybe_unused]] constexpr bool send_type_is_input_parameter = !has_to_be_computed<decltype(send_type)>;
        [[maybe_unused]] constexpr bool recv_type_is_input_parameter = !has_to_be_computed<decltype(recv_type)>;

        KASSERT(
            // if the send type is user provided, kamping no longer can deduce the number of elements to send from the
            // size of the recv buffer
            send_type_is_input_parameter || is_same_on_all_ranks(send_buf.size()),
            "All PEs have to send the same number of elements. Use allgatherv, if you want to send a different number "
            "of "
            "elements.",
            assert::light_communication
        );

        // get the send counts
        using default_send_count_type = decltype(kamping::send_count_out());
        auto send_count =
            internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
                std::tuple(),
                args...
            )
                .construct_buffer_or_rebind();
        constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
        if constexpr (do_compute_send_count) {
            send_count.underlying() = asserting_cast<int>(send_buf.size());
        }

        // get the receive counts
        using default_recv_count_type = decltype(kamping::recv_count_out());
        auto recv_count =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_count, default_recv_count_type>(
                std::tuple(),
                args...
            )
                .construct_buffer_or_rebind();
        constexpr bool do_compute_recv_count = internal::has_to_be_computed<decltype(recv_count)>;
        if constexpr (do_compute_recv_count) {
            recv_count.underlying() = send_count.get_single_element();
        }

        auto compute_required_recv_buf_size = [&]() {
            return asserting_cast<size_t>(recv_count.get_single_element()) * size();
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            // if the recv type is user provided, kamping cannot make any assumptions about the required size of the
            // recv buffer
            recv_type_is_input_parameter || recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        // error code can be unused if KTHROW is removed at compile time
        [[maybe_unused]] int err = MPI_Allgather(
            send_buf.data(),
            send_count.get_single_element(),
            send_type.get_single_element(),
            recv_buf.data(),
            recv_count.get_single_element(),
            recv_type.get_single_element(),
            this->mpi_communicator()
        );
        this->mpi_error_hook(err, "MPI_Allgather");

        return make_mpi_result<std::tuple<Args...>>(
            std::move(recv_buf),
            std::move(send_count),
            std::move(recv_count),
            std::move(send_type),
            std::move(recv_type)
        );
    }
}

/// @brief Wrapper for the in-place version of \c MPI_Allgather.
///
/// This variant must be called collectively by all ranks in the communicator.
///
/// The following parameters are required:
/// - \ref kamping::send_recv_buf() containing the data that is sent to the root. Opposed to the non-inplace version,
/// this is required to already have size `size() * send_recv_count` and the data contributed by each rank is already at
/// the correct location in the buffer.
///
/// The following parameters are optional:
/// - \ref kamping::send_recv_count() specifying how many elements are sent and received. If omitted, the size
/// `send_recv_buf.size() / size()` is used.
///
/// - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as send and recv type. If omitted, the \c MPI
/// datatype is derived automatically based on send_recv_buf's underlying \c value_type.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgather_inplace(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_recv_buf),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_count, send_recv_type)
    );

    // get the send/recv buffer and type
    auto buffer =
        internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...).construct_buffer_or_rebind();
    using value_type = typename std::remove_reference_t<decltype(buffer)>::value_type;

    auto type = internal::determine_mpi_send_recv_datatype<value_type, decltype(buffer)>(args...);
    [[maybe_unused]] constexpr bool type_is_input_parameter = !has_to_be_computed<decltype(type)>;

    KASSERT(
        // if the type is user provided, kamping no longer can deduce the number of elements to send from the size
        // of the recv buffer
        type_is_input_parameter || is_same_on_all_ranks(buffer.size()),
        "All PEs have to send the same number of elements. Use allgatherv, if you want to send a different number of "
        "elements.",
        assert::light_communication
    );

    // get the send counts
    using default_count_type = decltype(kamping::send_recv_count_out());
    auto count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_count, default_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();
    constexpr bool do_compute_count = internal::has_to_be_computed<decltype(count)>;
    // Opposed the non-inplace version, this requires that the data contributed by each rank is already at the correct
    // location in the buffer. Therefore the count is inferred as buffer.size() / size() if not given.
    KASSERT(
        (!do_compute_count || buffer.size() % size() == 0lu),
        "There is no send_recv_count given and the number of elements in send_recv_buf is not divisible by the number "
        "of "
        "ranks "
        "in the communicator.",
        assert::light
    );
    if constexpr (do_compute_count) {
        count.underlying() = asserting_cast<int>(buffer.size() / size());
    }

    auto compute_required_buf_size = [&]() {
        return asserting_cast<size_t>(count.get_single_element()) * size();
    };
    buffer.resize_if_requested(compute_required_buf_size);
    KASSERT(
        // if the type is user provided, kamping cannot make any assumptions about the required size of the buffer
        type_is_input_parameter || buffer.size() >= compute_required_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgather(
        MPI_IN_PLACE,               // sendbuf
        0,                          // sendcount (ignored)
        MPI_DATATYPE_NULL,          // sendtype (ignored)
        buffer.data(),              // recvbuf
        count.get_single_element(), // recvcount
        type.get_single_element(),  // recvtype
        this->mpi_communicator()    // communicator
    );
    this->mpi_error_hook(err, "MPI_Allgather");

    return make_mpi_result<std::tuple<Args...>>(std::move(buffer), std::move(count), std::move(type));
}

/// @brief Wrapper for \c MPI_Allgatherv.
///
/// This wrapper for \c MPI_Allgatherv collects possibly different amounts of data from each rank to all ranks. It is
/// semantically equivalent to performing a \c gatherv() followed by a broadcast of the collected data.
///
/// The following parameters are required:
/// - kamping::send_buf() containing the data that is sent to all other ranks.
///
/// The following parameters are optional but result in communication overhead if omitted:
/// - \ref kamping::recv_counts() containing the number of elements to receive from each rank. This parameter is
/// mandatory if \ref kamping::recv_type() is given.
///
/// The following parameters are optional:
/// - kamping::send_count() specifying how many elements are sent. If
/// omitted, the size of the send buffer is used. This parameter is mandatory if \ref kamping::send_type() is given.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c
/// MPI datatype is derived automatically based on send_buf's underlying \c value_type.
///
/// - kamping::recv_buf() specifying a buffer for the output.  Afterwards, this buffer will contain
/// all data from all send buffers. This requires a size of the underlying storage of at least  `max(recv_counts[i] +
/// recv_displs[i])` for \c i in `[0, communicator size)`.
///
/// - kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]`
/// elements starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If omitted, this is calculated as
/// the exclusive prefix-sum of `recv_counts`.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c
/// MPI datatype is derived automatically based on recv_buf's underlying \c value_type.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgatherv(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(send_count, recv_buf, recv_counts, recv_displs, send_type, recv_type)
    );

    // get send_buf
    auto send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

    // get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    // get send/recv types
    auto [send_type, recv_type] =
        internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_type_is_input_parameter = !internal::has_to_be_computed<decltype(send_type)>;
    [[maybe_unused]] constexpr bool recv_type_is_input_parameter = !internal::has_to_be_computed<decltype(recv_type)>;

    // get the send counts
    using default_send_count_type = decltype(kamping::send_count_out());
    auto send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();
    constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
    if constexpr (do_compute_send_count) {
        send_count.underlying() = asserting_cast<int>(send_buf.size());
    }
    // get the recv counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");
    // calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_counts),
        "Receive counts are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_counts) {
        recv_counts.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        this->allgather(
            kamping::send_buf(static_cast<int>(send_count.get_single_element())),
            kamping::recv_buf(recv_counts.get())
        );
    } else {
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
    }

    // Get recv_displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
    auto recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));

    // Calculate recv_displs if necessary
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_displs),
        "Receive displacements are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_displs) {
        recv_displs.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_displs.size() >= this->size(), "Recv displs buffer is not large enough.", assert::light);
        std::exclusive_scan(recv_counts.data(), recv_counts.data() + this->size(), recv_displs.data(), 0);
    } else {
        KASSERT(recv_displs.size() >= this->size(), "Recv displs buffer is not large enough.", assert::light);
    }

    auto compute_required_recv_buf_size = [&]() {
        return compute_required_recv_buf_size_in_vectorized_communication(recv_counts, recv_displs, this->size());
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        // if the recv type is user provided, kamping cannot make any assumptions about the required size of the recv
        // buffer
        recv_type_is_input_parameter || recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgatherv(
        send_buf.data(),                 // sendbuf
        send_count.get_single_element(), // sendcount
        send_type.get_single_element(),  // sendtype
        recv_buf.data(),                 // recvbuf
        recv_counts.data(),              // recvcounts
        recv_displs.data(),              // recvdispls
        recv_type.get_single_element(),  // recvtype
        this->mpi_communicator()         // communicator
    );
    this->mpi_error_hook(err, "MPI_Allgatherv");

    return make_mpi_result<std::tuple<Args...>>(
        std::move(recv_buf),
        std::move(send_count),
        std::move(recv_counts),
        std::move(recv_displs),
        std::move(send_type),
        std::move(recv_type)
    );
}
/// @}
