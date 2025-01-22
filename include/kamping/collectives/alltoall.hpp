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

#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Alltoall.
///
/// This wrapper for \c MPI_Alltoall sends the same amount of data from each rank to each rank. The following
/// buffers are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank and divisible by the size of the communicator unless a send_count or a send_type is explicitly given
/// as parameter. Each rank receives the same number of elements from this buffer.
///
/// The following parameters are optional:
/// - \ref kamping::send_count() specifying how many elements are sent. If
/// omitted, the size of send buffer divided by communicator size is used.
/// This parameter is mandatory if \ref kamping::send_type() is given.
///
/// - \ref kamping::recv_count() specifying how many elements are received. If
/// omitted, the value of send_counts will be used.
/// This parameter is mandatory if \ref kamping::recv_type() is given.
///
/// - \ref kamping::recv_buf() specifying a buffer for the output. A buffer of at least
/// `recv_count * communicator size` is required.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype
/// is derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c MPI datatype
/// is derived automatically based on recv_buf's underlying \c value_type.
///
/// Inplace alltoall is supported by passing send_recv_buf as parameter. This changes the requirements for the other
/// parameters, see \ref Communicator::alltoall_inplace.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output parameters to be returned by value.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoall(Args... args) const {
    using namespace internal;
    constexpr bool inplace = internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    if constexpr (inplace) {
        return this->alltoall_inplace(std::forward<Args>(args)...);
    } else {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf),
            KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_count, recv_count, send_type, recv_type)
        );

        // Get the buffers
        auto const send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        using default_recv_value_type = std::remove_const_t<send_value_type>;

        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

        static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

        auto [send_type, recv_type] =
            internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
        [[maybe_unused]] constexpr bool recv_type_has_to_be_deduced = has_to_be_computed<decltype(recv_type)>;

        // Get the send counts
        using default_send_count_type = decltype(kamping::send_count_out());
        auto send_count =
            internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
                std::tuple(),
                args...
            )
                .construct_buffer_or_rebind();
        constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
        if constexpr (do_compute_send_count) {
            send_count.underlying() = asserting_cast<int>(send_buf.size() / size());
        }
        // Get the recv counts
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

        KASSERT(
            (!do_compute_send_count || send_buf.size() % size() == 0lu),
            "There are no send counts given and the number of elements in send_buf is not divisible by the number of "
            "ranks "
            "in the communicator.",
            assert::light
        );

        auto compute_required_recv_buf_size = [&]() {
            return asserting_cast<size_t>(recv_count.get_single_element()) * size();
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            // if the recv type is user provided, kamping cannot make any assumptions about the required size of the
            // recv buffer
            !recv_type_has_to_be_deduced || recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        // These KASSERTs are required to avoid a false warning from g++ in release mode
        KASSERT(send_buf.data() != nullptr, assert::light);
        KASSERT(recv_buf.data() != nullptr, assert::light);

        [[maybe_unused]] int err = MPI_Alltoall(
            send_buf.data(),                 // send_buf
            send_count.get_single_element(), // send_count
            send_type.get_single_element(),  // send_type
            recv_buf.data(),                 // recv_buf
            recv_count.get_single_element(), // recv_count
            recv_type.get_single_element(),  // recv_type
            mpi_communicator()               // comm
        );

        this->mpi_error_hook(err, "MPI_Alltoall");
        return make_mpi_result<std::tuple<Args...>>(
            std::move(recv_buf),   // recv_buf
            std::move(send_count), // send_count
            std::move(recv_count), // recv_count
            std::move(send_type),  // send_type
            std::move(recv_type)   // recv_type
        );
    }
}

/// @brief Wrapper for the in-place version of \ref Communicator::alltoall.
///
/// This variant must be called collectively by all ranks in the communicator. It sends the same amount of data from
/// each rank to each rank, using the same buffer for sending and receiving data.
///
/// The following parameteres are required:
///
/// - \ref kamping::send_recv_buf() containing the data that is sent to each rank and received from each rank. The size
/// of this buffer has to be the same at each rank and divisible by the size of the communicator unless a
/// send_recv_count or a send_recv_type is explicitly given as parameter. Each rank receives the same number of elements
/// from this buffer.
///
/// The following parameters are optional:
///
/// - \ref kamping::send_recv_count() specifying how many elements are sent and received. If
/// omitted, the size of send_recv_buf divided by communicator size is used.
/// This parameter is mandatory if \ref kamping::send_recv_type() is given.
/// -
/// \ref kamping::send_recv_type() specifying the \c MPI datatype to use as send and recv type. If omitted, the \c MPI
/// datatype is derived automatically based on send_recv_buf's underlying \c value_type.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output parameters to be returned by value.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoall_inplace(Args... args) const {
    using namespace internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_recv_buf),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_count, send_recv_type)
    );

    auto send_recv_buf =
        internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...).construct_buffer_or_rebind();
    using send_recv_value_type = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    auto send_recv_type =
        internal::determine_mpi_send_recv_datatype<send_recv_value_type, decltype(send_recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_recv_type_has_to_be_deduced = has_to_be_computed<decltype(send_recv_type)>;

    // Get the optional recv_count parameter. If the parameter is not given, allocate a new container.
    using default_count_type = decltype(kamping::send_recv_count_out());
    auto count_param = internal::select_parameter_type_or_default<ParameterType::send_recv_count, default_count_type>(
                           std::tuple(),
                           args...
    )
                           .construct_buffer_or_rebind();
    constexpr bool count_has_to_be_computed = has_to_be_computed<decltype(count_param)>;

    KASSERT(
        (!count_has_to_be_computed || send_recv_buf.size() % size() == 0lu),
        "There is no send_recv_count given and the number of elements in send_recv_buf is not divisible by the number "
        "of "
        "ranks "
        "in the communicator.",
        assert::light
    );

    if constexpr (count_has_to_be_computed) {
        count_param.underlying() = asserting_cast<int>(send_recv_buf.size() / size());
    }
    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(count_param.get_single_element()) * size();
    };
    send_recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        // if the type is user provided, kamping cannot make any assumptions about the required size of the
        // buffer
        !send_recv_type_has_to_be_deduced || send_recv_buf.size() >= compute_required_recv_buf_size(),
        "send_recv_buf is not large enough to hold all received elements.",
        assert::light
    );
    int err = MPI_Alltoall(
        MPI_IN_PLACE,                        // send_buf
        0,                                   // send_count (ignored)
        MPI_DATATYPE_NULL,                   // send_type (ignored)
        send_recv_buf.data(),                // recv_buf
        count_param.get_single_element(),    // recv_count
        send_recv_type.get_single_element(), // recv_type
        mpi_communicator()                   // comm
    );
    this->mpi_error_hook(err, "MPI_Alltoall");

    return make_mpi_result<std::tuple<Args...>>(
        std::move(send_recv_buf),
        std::move(count_param),
        std::move(send_recv_type)
    );
}

/// @brief Wrapper for \c MPI_Alltoallv.
///
/// This wrapper for \c MPI_Alltoallv sends the different amounts of data from each rank to each rank. The following
/// buffers are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at least
/// the sum of the send_counts argument.
///
/// - \ref kamping::send_counts() containing the number of elements to send to each rank.
///
/// The following parameters are optional but result in communication overhead if omitted:
/// - \ref kamping::recv_counts() containing the number of elements to receive from each rank.
/// This parameter is mandatory if \ref kamping::recv_type() is given.
///
/// The following buffers are optional:
/// - \ref kamping::recv_buf() specifying a buffer for the output. Afterwards, this buffer will contain
/// the data received as specified for send_buf. A buffer size of at least  `max(recv_counts[i] +
/// recv_displs[i])` for \c i in `[0, communicator size)` elements is required.
///
/// - \ref kamping::send_displs() containing the offsets of the messages in send_buf. The `send_counts[i]` elements
/// starting at `send_buf[send_displs[i]]` will be sent to rank `i`. If omitted, this is calculated as the exclusive
/// prefix-sum of `send_counts`.
///
/// - \ref kamping::recv_displs() containing the offsets of the messages in recv_buf. The `recv_counts[i]` elements
/// starting at `recv_buf[recv_displs[i]]` will be received from rank `i`. If omitted, this is calculated as the
/// exclusive prefix-sum of `recv_counts`.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c MPI datatype is
/// derived automatically based on recv_buf's underlying \c value_type.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
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
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoallv(Args... args) const {
    // Get all parameter objects
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
        KAMPING_OPTIONAL_PARAMETERS(recv_counts, recv_buf, send_displs, recv_displs, send_type, recv_type)
    );

    // Get send_buf
    auto const& send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    // Get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    // Get send/recv types
    auto [send_type, recv_type] =
        internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_type_has_to_be_deduced = internal::has_to_be_computed<decltype(send_type)>;
    [[maybe_unused]] constexpr bool recv_type_has_to_be_deduced = internal::has_to_be_computed<decltype(recv_type)>;

    // Get send_counts
    auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                  .template construct_buffer_or_rebind<DefaultContainerType>();
    using send_counts_type = typename std::remove_reference_t<decltype(send_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_counts_type>, int>, "Send counts must be of type int");
    static_assert(
        !internal::has_to_be_computed<decltype(send_counts)>,
        "Send counts must be given as an input parameter"
    );
    KASSERT(send_counts.size() >= this->size(), "Send counts buffer is not large enough.", assert::light);

    // Get recv_counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");

    // Get send_displs
    using default_send_displs_type = decltype(kamping::send_displs_out(alloc_new<DefaultContainerType<int>>));
    auto send_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using send_displs_type = typename std::remove_reference_t<decltype(send_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_displs_type>, int>, "Send displs must be of type int");

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

    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

    // Calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_recv_counts),
        "Receive counts are given on some ranks and have to be computed on others",
        assert::light_communication
    );
    if constexpr (do_calculate_recv_counts) {
        /// @todo make it possible to test whether this additional communication is skipped
        recv_counts.resize_if_requested([&]() { return this->size(); });
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        this->alltoall(kamping::send_buf(send_counts.get()), kamping::recv_buf(recv_counts.get()));
    } else {
        KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
    }

    // Calculate send_displs if necessary
    constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;
    KASSERT(
        is_same_on_all_ranks(do_calculate_send_displs),
        "Send displacements are given on some ranks and have to be computed on others",
        assert::light_communication
    );

    if constexpr (do_calculate_send_displs) {
        send_displs.resize_if_requested([&]() { return this->size(); });
        KASSERT(send_displs.size() >= this->size(), "Send displs buffer is not large enough.", assert::light);
        std::exclusive_scan(send_counts.data(), send_counts.data() + this->size(), send_displs.data(), 0);
    } else {
        KASSERT(send_displs.size() >= this->size(), "Send displs buffer is not large enough.", assert::light);
    }

    // Check that send displs and send counts are large enough
    KASSERT(
        // if the send type is user provided, kamping cannot make any assumptions about the size of the send
        // buffer
        !send_type_has_to_be_deduced
            || *(send_counts.data() + this->size() - 1) +       // Last element of send_counts
                       *(send_displs.data() + this->size() - 1) // Last element of send_displs
                   <= asserting_cast<int>(send_buf.size()),
        assert::light
    );

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
        !recv_type_has_to_be_deduced || recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // Do the actual alltoallv
    [[maybe_unused]] int err = MPI_Alltoallv(
        send_buf.data(),                // send_buf
        send_counts.data(),             // send_counts
        send_displs.data(),             // send_displs
        send_type.get_single_element(), // send_type
        recv_buf.data(),                // send_counts
        recv_counts.data(),             // recv_counts
        recv_displs.data(),             // recv_displs
        recv_type.get_single_element(), // recv_type
        mpi_communicator()              // comm
    );

    this->mpi_error_hook(err, "MPI_Alltoallv");

    return internal::make_mpi_result<std::tuple<Args...>>(
        std::move(recv_buf),    // recv_buf
        std::move(recv_counts), // recv_counts
        std::move(recv_displs), // recv_displs
        std::move(send_displs), // send_displs
        std::move(send_type),   // send_type
        std::move(recv_type)    // recv_type
    );
}
/// @}
