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

/// @file
/// @brief Plugin to dispatch to one of multiple possible algorithms for alltoallv exchanges.

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/plugin/alltoall_grid.hpp"
#include "kamping/plugin/alltoall_sparse.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

#pragma once

namespace kamping::plugin {

namespace dispatch_alltoall {

/// @brief Parameter types used for the DispatchAlltoall plugin.
enum class ParameterType {
    comm_volume_threshold ///< Tag used to the communication volume threshold to use within alltoall_dispatch.
};

/// @brief The threshold for the maximum bottleneck communication volume in number of bytes indicating for when to
/// switch from grid to builtin alltoall.
/// @param num_bytes Threshold volume in number of bytes.
/// @return The corresponding parameter object.
inline auto comm_volume_threshold(size_t num_bytes) {
    return internal::make_data_buffer<
        ParameterType,
        ParameterType::comm_volume_threshold,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        size_t>(std::move(num_bytes));
}

//
namespace internal {
/// @brief Predicate to check whether an argument provided to alltoallv_dispatch shall be discarded in the internal
/// calls.
struct PredicateDispatchAlltoall {
    /// @brief Function to check whether an argument provided to \ref DispatchAlltoall::alltoallv_dispatch() shall be
    /// discarded in the send call.
    ///
    /// @tparam Arg Argument to be checked.
    /// @return \c True (i.e. discard) iff Arg's parameter_type is `volume_threshold`, `send_counts`.
    template <typename Arg>
    static constexpr bool discard() {
        using ptypes_to_ignore = kamping::internal::type_list<
            std::integral_constant<
                dispatch_alltoall::ParameterType,
                dispatch_alltoall::ParameterType::comm_volume_threshold>,
            std::integral_constant<kamping::internal::ParameterType, kamping::internal::ParameterType::send_counts>>;
        using ptype_entry =
            std::integral_constant<kamping::internal::parameter_type_t<Arg>, kamping::internal::parameter_type_v<Arg>>;
        return ptypes_to_ignore::contains<ptype_entry>;
    }
};

} // namespace internal

} // namespace dispatch_alltoall
/// @brief Plugin providing an alltoallv exchange method which calls one of multiple underlying alltoallv exchange
/// algorithms depending on the communication volume.
/// @see \ref DispatchAlltoall::alltoallv_dispatch() for more information.
template <typename Comm, template <typename...> typename DefaultContainerType>
class DispatchAlltoall : public plugin::PluginBase<Comm, DefaultContainerType, DispatchAlltoall> {
public:
    /// @brief Alltoallv exchange method which uses the communication volume to either exchange the data using
    /// GridCommunicator::alltoallv() (latency in about sqrt(comm.size())) or builtin MPI_Alltoallv (potentially linear
    /// latency).
    ///
    /// If the bottleneck send communication volume on all ranks is smaller than a given threshold (in number bytes),
    /// our grid alltoall communication is used. Otherwise we use the builtin MPI alltoallv exchange.
    ///
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at
    /// least the sum of the send_counts argument.
    ///
    /// - \ref kamping::send_counts() containing the number of elements to send to each rank.
    ///
    /// The following buffers are optional:
    /// - \ref dispatch_alltoall::comm_volume_threshold() containing the threshold for the maximum bottleneck
    /// communication volume in bytes indicating to switch from grid to builtin alltoall exchange. If ommitted, a
    /// threshold value of 2000 bytes is used.
    /// - \ref kamping::recv_counts() containing the number of elements to receive from each rank.
    /// This parameter is mandatory if \ref kamping::recv_type() is given.
    ///
    /// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
    /// the data received as specified for send_buf. The buffer will be resized according to the buffer's
    /// kamping::BufferResizePolicy. If resize policy is kamping::BufferResizePolicy::no_resize, the buffer's underlying
    /// storage must be large enough to store all received elements.
    ///
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional parameters described above.
    template <typename... Args>
    auto alltoallv_dispatch(Args... args) const {
        [[maybe_unused]] auto& self     = this->to_communicator();
        auto&                  send_buf = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
        using send_value_type           = typename std::remove_reference_t<decltype(send_buf)>::value_type;

        // get send counts (needed for algorithm dispatch)
        auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();

        // get send communication volume threshold for when to switch from grid to builtin alltoall
        using volume_threshold_param_type = std::integral_constant<
            dispatch_alltoall::ParameterType,
            dispatch_alltoall::ParameterType::comm_volume_threshold>;
        constexpr size_t volume_threshold_default_value = 2000;
        using default_comm_volume_threshold_type =
            decltype(dispatch_alltoall::comm_volume_threshold(volume_threshold_default_value));
        auto&& volume_threshold =
            internal::select_parameter_type_or_default<volume_threshold_param_type, default_comm_volume_threshold_type>(
                std::tuple(volume_threshold_default_value),
                args...
            );

        size_t const max_bottleneck_send_volume =
            self.allreduce_single(kamping::send_buf(send_buf.size()), op(ops::max<size_t>{}));

        /// remove comm_volume_threshold and unpacked send_counts from caller provided argument list before forwarding
        /// it underlying allotall exchanges
        auto const filter_args = [&]() {
            return filter_args_into_tuple<dispatch_alltoall::internal::PredicateDispatchAlltoall>(args...);
        };

        if (max_bottleneck_send_volume * sizeof(send_value_type) < volume_threshold.get_single_element()) {
            // max bottleneck send volume is small ==> use grid exchange
            auto callable = [&](auto... argsargs) {
                initialize();
                return _grid_communicator.value().alltoallv(kamping::send_counts(send_counts), std::move(argsargs)...);
            };
            return std::apply(callable, filter_args());
        }

        // otherwise resort to builtin MPI_Alltoallv.
        auto callable = [&](auto... argsargs) {
            return self.alltoallv(kamping::send_counts(send_counts), std::move(argsargs)...);
        };
        return std::apply(callable, filter_args());
    }

    /// @brief Initialized the grid communicator. If not explicitly called by the user this will be done during the
    /// first call to alltoallv_dispatch which internally uses grid communication.
    void initialize() const {
        if (_grid_communicator.has_value()) {
            return;
        } else {
            _grid_communicator = std::make_optional(this->to_communicator().make_grid_communicator());
        }
    }

private:
    mutable std::optional<grid::GridCommunicator<DefaultContainerType>>
        _grid_communicator; ///< Grid communicator to use for grid alltoall exchange.
};
} // namespace kamping::plugin
