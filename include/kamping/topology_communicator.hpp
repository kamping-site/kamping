// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <kamping/collectives/collectives_helpers.hpp>
#include <kamping/communicator.hpp>

namespace kamping {
/// @brief Wrapper for an MPI communicator with topology providing access to \c rank() and \c size() of the
/// communicator. The \ref Communicator is also access point to all MPI communications provided by KaMPIng.
/// @tparam DefaultContainerType The default container type to use for containers created by KaMPIng. Defaults to
/// std::vector.
/// @tparam Plugins Plugins adding functionality to KaMPIng. Plugins should be classes taking a ``Communicator``
/// template parameter and can assume that they are castable to `Communicator` from which they can
/// call any function of `kamping::Communicator`. See `test/plugin_tests.cpp` for examples.
template <
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename>
    typename... Plugins>
class TopologyCommunicator
    : public Communicator<DefaultContainerType>,
      public Plugins<TopologyCommunicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    using Communicator<DefaultContainerType>::Communicator;
    /// @brief Type of the default container type to use for containers created inside operations of this communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    template <typename... Args>
    auto neighbor_alltoall(Args... args) const {
        using namespace internal;
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf),
            KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_count, recv_count, send_type, recv_type)
        );
        // Get the buffers
        auto const&& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        using default_recv_value_type = std::remove_const_t<send_value_type>;

        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

        static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

        auto&& [send_type, recv_type] =
            internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
        [[maybe_unused]] constexpr bool recv_type_has_to_be_deduced = has_to_be_computed<decltype(recv_type)>;

        // Get the send counts
        using default_send_count_type = decltype(kamping::send_count_out());
        auto&& send_count =
            internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
                std::tuple(),
                args...
            )
                .construct_buffer_or_rebind();
        constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
        if constexpr (do_compute_send_count) {
            send_count.underlying() = asserting_cast<int>(send_buf.size() / this->size());
        }
        // Get the recv counts
        using default_recv_count_type = decltype(kamping::recv_count_out());
        auto&& recv_count =
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
            (!do_compute_send_count || send_buf.size() % this->size() == 0lu),
            "There are no send counts given and the number of elements in send_buf is not divisible by the number "
            "of "
            "ranks "
            "in the communicator.",
            assert::light
        );

        auto compute_required_recv_buf_size = [&]() {
            return asserting_cast<size_t>(recv_count.get_single_element()) * this->size();
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

        [[maybe_unused]] int err = MPI_Neighbor_alltoall(
            send_buf.data(),                 // send_buf
            send_count.get_single_element(), // send_count
            send_type.get_single_element(),  // send_type
            recv_buf.data(),                 // recv_buf
            recv_count.get_single_element(), // recv_count
            recv_type.get_single_element(),  // recv_type
            this->mpi_communicator()         // comm
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
};
} // namespace kamping
