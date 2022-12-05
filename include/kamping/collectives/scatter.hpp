// This file is part of KaMPIng
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

#include <numeric>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

namespace {
// Broadcasts a value from one PE to all PEs.
template <typename T, template <typename...> typename DefaultContainerType>
int bcast_value(kamping::Communicator<DefaultContainerType> const& comm, T const bcast_value, int const root) {
    using namespace kamping::internal;
    using namespace kamping;
    T                          bcast_result = bcast_value;
    [[maybe_unused]] int const result = MPI_Bcast(&bcast_result, 1, mpi_datatype<T>(), root, comm.mpi_communicator());
    THROW_IF_MPI_ERROR(result, MPI_Bcast);
    return bcast_result;
}
} // anonymous namespace

/// @brief Wrapper for \c MPI_Scatter.
///
/// This wrapper for \c MPI_Scatter distributes data on the root PE evenly across all PEs in the current
/// communicator.
///
/// The following parameters are mandatory:
/// - \ref kamping::send_buf() containing the data to be evenly distributed across all PEs. The size of
/// this buffer must be divisible by the number of PEs in the current communicator. Non-root PEs can omit a send
/// buffer by passing `kamping::ignore` to \ref kamping::send_buf().
///
/// The following parameters are optional but incur communication overhead if omitted:
/// - \ref kamping::recv_counts() specifying the number of elements sent to each PE. If this parameter is omitted,
/// the number of elements sent to each PE is computed based on the size of the \ref kamping::send_buf() on the root
/// PE and broadcasted to other PEs.
///
/// The following parameters are optional:
/// - \ref kamping::root() specifying the rank of the root PE. If omitted, the default root PE of the communicator
/// is used instead.
/// - \ref kamping::recv_buf() containing the received data. If omitted, a new buffer is allocated and returned.
///
/// @tparam Args Deduced template parameters.
/// @param args Required and optionally optional parameters.
/// @return kamping::MPIResult wrapping the output buffer if not specified as an input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, plugins...>::scatter(Args... args) const {
    using namespace kamping::internal;

    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(root, recv_buf, recv_counts)
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

    // Mandatory parameter send_buf()
    auto send_buf              = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
    auto const*  send_buf_ptr  = send_buf.data();
    KASSERT(!is_root(int_root) || send_buf_ptr != nullptr, "Send buffer must be specified on root.", assert::light);

    // Compute sendcount based on the size of the sendbuf
    KASSERT(
        send_buf.size() % size() == 0u,
        "Size of the send buffer (" << send_buf.size() << ") is not divisible by the number of PEs (" << size()
                                    << ") in the communicator."
    );
    int const send_count = asserting_cast<int>(send_buf.size() / size());

    // Optional parameter: recv_buf()
    // Default: allocate new container
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<DefaultContainerType<send_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    // Make sure that send and recv buffers use the same type
    static_assert(
        std::is_same_v<send_value_type, recv_value_type>,
        "Mismatching send_buf() and recv_buf() value types."
    );

    // Optional parameter: recv_count()
    // Default: compute value based on send_buf.size on root
    using default_recv_count_type = decltype(kamping::recv_counts_out(NewContainer<int>{}));
    auto&& recv_count_param =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );
    constexpr bool is_output_parameter = has_to_be_computed<decltype(recv_count_param)>;

    KASSERT(
        is_same_on_all_ranks(is_output_parameter),
        "recv_count() parameter is an output parameter on some PEs, but not on alle PEs.",
        assert::light_communication
    );

    // If it is an output parameter, broadcast send_count to get recv_count
    static_assert(
        std::remove_reference_t<decltype(recv_count_param)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );
    if constexpr (is_output_parameter) {
        recv_count_param.underlying() = bcast_value(*this, send_count, int_root);
    }

    int recv_count = recv_count_param.get_single_element();

    // Validate against send_count
    KASSERT(
        recv_count == bcast_value(*this, send_count, int_root),
        "Specified recv_count() does not match the send count.",
        assert::light_communication
    );

    recv_buf.resize(static_cast<std::size_t>(recv_count));
    auto* recv_buf_ptr = recv_buf.data();

    [[maybe_unused]] int const err = MPI_Scatter(
        send_buf_ptr,
        send_count,
        mpi_send_type,
        recv_buf_ptr,
        recv_count,
        mpi_recv_type,
        int_root,
        mpi_communicator()
    );
    THROW_IF_MPI_ERROR(err, MPI_Scatter);

    return make_mpi_result(std::move(recv_buf), std::move(recv_count_param));
}

/// @brief Wrapper for \c MPI_Scatterv.
///
/// This wrapper for \c MPI_Scatterv distributes data on the root PE across all PEs in the current communicator.
///
/// The following parameters are mandatory:
/// - \ref kamping::send_buf() [on all PEs] containing the data to be distributed across all PEs. Non-root PEs can omit
/// a send buffer by passing `kamping::ignore` to \ref kamping::send_buf().
///
/// Of the following parameters, one can be omitted at the cost of communication overhead (1x MPI_Scatter or 1x
/// MPI_Gather). Provide both parameters to avoid overheads.
/// - \ref kamping::send_counts() [on root PE] specifying the number of elements sent to each PE. If this parameter is
/// omitted, the number of elements sent to each PE is computed based on the provided \ref kamping::recv_counts() on
/// other PEs.
/// - \ref kamping::recv_counts() [on all PEs] specifying the number of elements sent to each PE. If this parameter is
/// omitted, the number of elements sent to each PE is computed based on \ref kamping::send_counts() provided on the
/// root PE.
///
/// The following parameter can be omitted at the cost of computational overhead:
/// - \ref kamping::send_displs() [on root PE] specifying the data displacements in the send buffer. If omitted, a new
/// buffer is allocated and displacements are computed based on the \ref kamping::send_counts().
///
/// The following parameters are optional:
/// - \ref kamping::root() [on all PEs] specifying the rank of the root PE. If omitted, the default root PE of the
/// communicator is used instead.
/// - \ref kamping::recv_buf() [on all PEs] containing the received data. If omitted, a new buffer is allocated and
/// returned.
///
/// @tparam Args Deduced template parameters.
/// @param args Required and optionally optional parameters.
/// @return kamping::MPIResult wrapping the output buffer if not specified as an input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, plugins...>::scatterv(Args... args) const {
    using namespace kamping::internal;

    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(root, send_counts, send_displs, recv_buf, recv_counts)
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

    // Mandatory parameter send_buf()
    auto send_buf              = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();
    auto const*  send_buf_ptr  = send_buf.data();
    KASSERT(!is_root(root_val) || send_buf_ptr != nullptr, "Send buffer must be specified on root.", assert::light);

    // Optional parameter: recv_buf()
    // Default: allocate new container
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<DefaultContainerType<send_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

    // Make sure that send and recv buffers use the same type
    static_assert(
        std::is_same_v<send_value_type, recv_value_type>,
        "Mismatching send_buf() and recv_buf() value types."
    );

    // Optional parameters: send_counts, send_displs, recv_counts
    using default_send_counts_type = decltype(send_counts_out(NewContainer<DefaultContainerType<int>>{}));
    auto&& send_counts_param =
        select_parameter_type_or_default<ParameterType::send_counts, default_send_counts_type>(std::tuple(), args...);

    using default_send_displs_type = decltype(send_displs_out(NewContainer<DefaultContainerType<int>>{}));
    auto&& send_displs_param =
        select_parameter_type_or_default<ParameterType::send_displs, default_send_displs_type>(std::tuple(), args...);

    using default_recv_counts_type = decltype(recv_counts_out());
    auto&& recv_counts_param =
        select_parameter_type_or_default<ParameterType::recv_counts, default_recv_counts_type>(std::tuple(), args...);

    // Check that recv_counts() can be used to compute send_counts(); or send_counts() is given on the root PE
    [[maybe_unused]] constexpr bool recv_counts_given = !has_to_be_computed<decltype(recv_counts_param)>;
    [[maybe_unused]] constexpr bool send_counts_given = !has_to_be_computed<decltype(send_counts_param)>;
    KASSERT(
        this->is_same_on_all_ranks(recv_counts_given),
        "recv_counts() must be given on all PEs or on no PEs",
        assert::light_communication
    );
    KASSERT(
        !is_root(root_val) || recv_counts_given || send_counts_given,
        "send_counts() must be given on the root PE; or recv_counts() must be given on all PEs",
        assert::light_communication
    );

    // Check the size of input parameters
    KASSERT(
        !is_root(root_val)
            || has_to_be_computed<decltype(send_counts_param)> || send_counts_param.get().size() >= size(),
        "send_counts() is smaller than the number of PEs",
        assert::light
    );
    KASSERT(
        !is_root(root_val)
            || has_to_be_computed<decltype(send_displs_param)> || send_displs_param.get().size() >= size(),
        "send_displs() is smaller than the number of PEs",
        assert::light
    );
    KASSERT(
        has_to_be_computed<decltype(recv_counts_param)> || recv_counts_param.get().size() >= 1,
        "recv_counts() may not be empty",
        assert::light
    );

    // Compute missing counts / displs parameters
    if constexpr (has_to_be_computed<decltype(send_counts_param)>) {
        gather(
            kamping::send_buf(recv_counts_param.underlying()),
            kamping::root(root_val),
            kamping::recv_buf(send_counts_param.underlying())
        );
    }

    if constexpr (has_to_be_computed<decltype(recv_counts_param)>) {
        scatter(
            kamping::send_buf(Span{send_counts_param.data(), size()}),
            kamping::root(root_val),
            kamping::recv_buf(recv_counts_param.underlying())
        );
    }

    if constexpr (has_to_be_computed<decltype(send_displs_param)>) {
        if (is_root(root_val)) {
            send_displs_param.resize(size());
            auto* send_displs_ptr = send_displs_param.data();
            auto* send_counts_ptr = send_counts_param.data();
            std::exclusive_scan(send_counts_ptr, send_counts_ptr + size(), send_displs_ptr, 0);
        }
    }

    recv_buf.resize(static_cast<std::size_t>(recv_counts_param.underlying()));
    [[maybe_unused]] int const err = MPI_Scatterv(
        send_buf_ptr,                   // send buffer
        send_counts_param.data(),       // send counts
        send_displs_param.data(),       // send displs
        mpi_send_type,                  // send type
        recv_buf.data(),                // recv buffer
        recv_counts_param.underlying(), // recv count
        mpi_recv_type,                  // recv type
        root_val,                       // root
        mpi_communicator()              // communicator
    );
    THROW_IF_MPI_ERROR(err, MPI_Scatterv);

    return make_mpi_result(
        std::move(recv_buf),
        std::move(recv_counts_param),
        std::move(send_counts_param),
        std::move(send_displs_param)
    );
}
