// This file is part of KaMPIng
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
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Scatter.
///
/// This wrapper for \c MPI_Scatter distributes data on the root PE evenly across all PEs in the current
/// communicator.
///
/// The following parameters are mandatory on the root rank:
/// - kamping::send_buf() containing the data to be evenly distributed across all PEs. The size of
/// this buffer must be divisible by the number of PEs in the current communicator. Non-root PEs can omit a send
/// buffer by passing `kamping::ignore<T>` as a parameter, or `T` as a template parameter to \ref kamping::send_buf().
///
/// The following parameters are optional but incur communication overhead if omitted:
/// - kamping::recv_counts() specifying the number of elements sent to each PE. If this parameter is omitted,
/// the number of elements sent to each PE is computed based on the size of the \ref kamping::send_buf() on the root
/// PE and broadcasted to other PEs.
///
/// The following parameters are optional:
/// - kamping::send_counts() specifying how many elements are sent to each process. This parameter has to be an integer.
/// If omitted, the size of send buffer divided by communicator size is used.
///
/// - kamping::root() specifying the rank of the root PE. If omitted, the default root PE of the communicator
/// is used instead.
///
/// - kamping::recv_buf() containing the received data. If omitted, a new buffer is allocated and returned.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::send_buf() and no \ref
/// kamping::recv_buf() is given.
/// @tparam Args Deduced template parameters.
/// @param args Required and optionally optional parameters.
/// @return kamping::MPIResult wrapping the output buffer if not specified as an input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::scatter(Args... args) const {
    using namespace kamping::internal;

    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(send_buf, send_counts, root, recv_buf, recv_counts)
    );

    // Optional parameter: root()
    // Default: communicator root
    using root_param_type = decltype(kamping::root(0));
    auto&& root_param =
        select_parameter_type_or_default<ParameterType::root, root_param_type>(std::tuple(root()), args...);
    int const int_root = root_param.rank_signed();
    KASSERT(
        is_valid_rank(int_root),
        "Invalid root rank " << int_root << " in communicator of size " << size(),
        assert::light
    );
    KASSERT(this->is_same_on_all_ranks(int_root), "Root has to be the same on all ranks.", assert::light_communication);

    // Parameter send_buf()
    using default_send_buf_type = decltype(kamping::send_buf(kamping::ignore<recv_value_type_tparam>));
    auto send_buf =
        select_parameter_type_or_default<ParameterType::send_buf, default_send_buf_type>(std::tuple(), args...).get();
    using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
    KASSERT(!is_root(int_root) || send_buf.data() != nullptr, "Send buffer must be specified on root.", assert::light);

    // Compute sendcount based on the size of the sendbuf
    using default_send_count_type = decltype(kamping::send_counts_out(alloc_new<int>));
    auto&& send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_send_count_type>(
            std::tuple(),
            args...
        );
    static_assert(
        std::remove_reference_t<decltype(send_count)>::is_single_element,
        "send_counts() parameter must be a single value."
    );
    constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
    if constexpr (do_compute_send_count) {
        KASSERT(
            !is_root(int_root) || send_buf.size() % size() == 0u,
            "No send count is given and the size of the send buffer ("
                << send_buf.size() << ") at the root is not divisible by the number of PEs (" << size()
                << ") in the communicator.",
            assert::light
        );
        send_count.underlying() = asserting_cast<int>(send_buf.size() / size());
    }

    // Optional parameter: recv_buf()
    // Default: allocate new container
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    static_assert(
        !std::is_same_v<recv_value_type, internal::unused_tparam>,
        "No send_buf or recv_buf parameter provided and no receive value given as template parameter. One of these is "
        "required."
    );

    // Make sure that send and recv buffers use the same type
    static_assert(
        std::is_same_v<send_value_type, recv_value_type> || std::is_same_v<send_value_type, internal::unused_tparam>,
        "Mismatching send_buf() and recv_buf() value types."
    );

    // Optional parameter: recv_count()
    // Default: compute value based on send_buf.size on root
    using default_recv_count_type = decltype(kamping::recv_counts_out(alloc_new<int>));
    auto&& recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );
    constexpr bool do_compute_recv_count = has_to_be_computed<decltype(recv_count)>;

    KASSERT(
        is_same_on_all_ranks(do_compute_recv_count),
        "recv_count() parameter is an output parameter on some PEs, but not on alle PEs.",
        assert::light_communication
    );

    // If it is an output parameter, broadcast send_count to get recv_count
    static_assert(
        std::remove_reference_t<decltype(recv_count)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );
    if constexpr (do_compute_recv_count) {
        recv_count.underlying() = send_count.get_single_element();
        this->bcast_single(send_recv_buf(recv_count.underlying()), kamping::root(int_root));
    }

    // Validate against send_count on root
    KASSERT(
        recv_count.get_single_element() ==
            [&]() {
                int send_count_on_root = send_count.get_single_element();
                this->bcast_single(send_recv_buf(send_count_on_root), kamping::root(int_root));
                return send_count_on_root;
            }(),
        "Specified recv_count() does not match the send count.",
        assert::light_communication
    );

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(recv_count.get_single_element());
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    [[maybe_unused]] int const err = MPI_Scatter(
        send_buf.data(),
        send_count.get_single_element(),
        mpi_send_type,
        recv_buf.data(),
        recv_count.get_single_element(),
        mpi_recv_type,
        int_root,
        mpi_communicator()
    );
    THROW_IF_MPI_ERROR(err, MPI_Scatter);

    return make_mpi_result(std::move(recv_buf), std::move(send_count), std::move(recv_count));
}

/// @brief Wrapper for \c MPI_Scatterv.
///
/// This wrapper for \c MPI_Scatterv distributes data on the root PE across all PEs in the current communicator.
///
/// The following parameters are mandatory on the root rank:
/// - kamping::send_buf() [on all PEs] containing the data to be distributed across all PEs. Non-root PEs can omit
/// a send buffer by passing `kamping::ignore<T>` as a parameter, or `T` as a template parameter to \ref
/// kamping::send_buf().
///
/// - kamping::send_counts() [on root PE] specifying the number of elements to send to each PE.
///
/// The following parameter can be omitted at the cost of communication overhead (1x MPI_Scatter)
/// - kamping::recv_counts() [on all PEs] specifying the number of elements sent to each PE. If this parameter is
/// omitted, the number of elements sent to each PE is computed based on kamping::send_counts() provided on the
/// root PE.
///
/// The following parameter can be omitted at the cost of computational overhead:
/// - kamping::send_displs() [on root PE] specifying the data displacements in the send buffer. If omitted, an exclusive
/// prefix sum of the send_counts is used.
///
/// The following parameters are optional:
/// - kamping::root() [on all PEs] specifying the rank of the root PE. If omitted, the default root PE of the
/// communicator is used instead.
/// - kamping::recv_buf() [on all PEs] containing the received data. The buffer will be resized according to the
/// buffer's kamping::BufferResizePolicy. If this is kamping::BufferResizePolicy::no_resize, the buffer's underlying
/// storage must be large enough to hold all received elements.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no kamping::send_buf() and no
/// kamping::recv_buf() is given.
/// @tparam Args Deduced template parameters.
/// @param args Required and optionally optional parameters.
/// @return kamping::MPIResult wrapping the output buffer if not specified as an input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::scatterv(Args... args) const {
    using namespace kamping::internal;

    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(send_buf, root, send_counts, send_displs, recv_buf, recv_counts)
    );

    // Optional parameter: root()
    // Default: communicator root
    using root_param_type = decltype(kamping::root(0));
    auto&& root_param =
        select_parameter_type_or_default<ParameterType::root, root_param_type>(std::tuple(root()), args...);
    int const root_val = root_param.rank_signed();
    KASSERT(
        is_valid_rank(root_val),
        "Invalid root rank " << root_val << " in communicator of size " << size(),
        assert::light
    );
    KASSERT(is_same_on_all_ranks(root_val), "Root has to be the same on all ranks.", assert::light_communication);

    // Parameter send_buf()
    using default_send_buf_type = decltype(kamping::send_buf(kamping::ignore<recv_value_type_tparam>));
    auto send_buf =
        select_parameter_type_or_default<ParameterType::send_buf, default_send_buf_type>(std::tuple(), args...).get();
    using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
    auto const*  send_buf_ptr  = send_buf.data();
    KASSERT(!is_root(root_val) || send_buf_ptr != nullptr, "Send buffer must be specified on root.", assert::light);

    // Optional parameter: recv_buf()
    // Default: allocate new container
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<send_value_type>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    static_assert(
        !std::is_same_v<recv_value_type, internal::unused_tparam>,
        "No send_buf or recv_buf parameter provided and no receive value given as template parameter. One of these is "
        "required."
    );

    // Make sure that send and recv buffers use the same type
    static_assert(
        std::is_same_v<send_value_type, recv_value_type> || std::is_same_v<send_value_type, internal::unused_tparam>,
        "Mismatching send_buf() and recv_buf() value types."
    );

    // Get send counts
    using default_send_counts_type = decltype(send_counts_out(alloc_new<DefaultContainerType<int>>));
    auto&& send_counts =
        select_parameter_type_or_default<ParameterType::send_counts, default_send_counts_type>(std::tuple(), args...);
    [[maybe_unused]] constexpr bool send_counts_provided = !has_to_be_computed<decltype(send_counts)>;
    KASSERT(
        !is_root(root_val) || send_counts_provided,
        "send_counts() must be given on the root PE.",
        assert::light_communication
    );
    KASSERT(
        !is_root(root_val) || send_counts.size() >= size(),
        "Send counts buffer is smaller than the number of PEs at the root PE.",
        assert::light
    );

    // Get send displacements
    using default_send_displs_type = decltype(send_displs_out(alloc_new<DefaultContainerType<int>>));
    auto&& send_displs =
        select_parameter_type_or_default<ParameterType::send_displs, default_send_displs_type>(std::tuple(), args...);

    constexpr bool do_compute_send_displs = has_to_be_computed<decltype(send_displs)>;
    if constexpr (do_compute_send_displs) {
        send_displs.resize_if_requested([&]() {
            if (is_root(root_val)) {
                return this->size();
            } else {
                // do not change the size of the buffer on non-root PEs
                return send_displs.size();
            }
        });
    }
    KASSERT(
        !is_root(root_val) || send_displs.size() >= size(),
        "Send displs buffer is smaller than the number of PEs at the root PE.",
        assert::light
    );
    if constexpr (do_compute_send_displs) {
        if (is_root(root_val)) {
            std::exclusive_scan(send_counts.data(), send_counts.data() + size(), send_displs.data(), 0);
        }
    }

    // Get recv counts
    using default_recv_counts_type = decltype(recv_counts_out(alloc_new<int>));
    auto&& recv_counts =
        select_parameter_type_or_default<ParameterType::recv_counts, default_recv_counts_type>(std::tuple(), args...);
    static_assert(
        std::remove_reference_t<decltype(recv_counts)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );

    // Check that recv_counts() can be used to compute send_counts(); or send_counts() is given on the root PE
    [[maybe_unused]] constexpr bool do_compute_recv_count = has_to_be_computed<decltype(recv_counts)>;
    KASSERT(
        this->is_same_on_all_ranks(do_compute_recv_count),
        "recv_counts() must be given on all PEs or on no PEs",
        assert::light_communication
    );

    if constexpr (do_compute_recv_count) {
        scatter(
            kamping::send_buf(send_counts.underlying()),
            kamping::root(root_val),
            kamping::recv_counts(1),
            kamping::recv_buf(recv_counts.underlying())
        );
    }

    auto compute_required_recv_buf_size = [&]() {
        return static_cast<size_t>(recv_counts.get_single_element());
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    [[maybe_unused]] int const err = MPI_Scatterv(
        send_buf_ptr,                     // send buffer
        send_counts.data(),               // send counts
        send_displs.data(),               // send displs
        mpi_send_type,                    // send type
        recv_buf.data(),                  // recv buffer
        recv_counts.get_single_element(), // recv count
        mpi_recv_type,                    // recv type
        root_val,                         // root
        mpi_communicator()                // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Scatterv);

    return make_mpi_result(std::move(recv_buf), std::move(recv_counts), std::move(send_counts), std::move(send_displs));
}
