
// This file is part of KaMPIng.
//
// Copyright 2023-2024 The KaMPIng Authors
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

#include <mpi.h>

#include "kamping/data_buffer.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/status.hpp"

namespace kamping {
namespace params {
/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Outputs the return status of the operation to the provided status object. The status object may be passed as
/// lvalue-reference or rvalue.
/// @tparam StatusObject type of the status object, may either be \c MPI_Status or \ref kamping::Status
/// @param status The status object.
template <typename StatusObject>
inline auto status_out(StatusObject&& status) {
    using status_type = std::remove_cv_t<std::remove_reference_t<StatusObject>>;
    static_assert(internal::type_list<MPI_Status, Status>::contains<status_type>);
    return internal::make_data_buffer_builder<
        internal::ParameterType::status,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        no_resize>(std::forward<StatusObject>(status));
}

/// @brief Constructs a status object internally, which may then be retrieved from \c kamping::MPIResult returned by the
/// operation.
inline auto status_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::status,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        no_resize>(alloc_new<Status>);
}

/// @brief pass \c MPI_STATUS_IGNORE to the underlying MPI call.
inline auto status(internal::ignore_t<void>) {
    return internal::
        make_empty_data_buffer_builder<Status, internal::ParameterType::status, internal::BufferType::ignore>();
}

/// @brief pass \c MPI_STATUSES_IGNORE to the underlying MPI call.
inline auto statuses(internal::ignore_t<void>) {
    return internal::
        make_empty_data_buffer_builder<MPI_Status, internal::ParameterType::statuses, internal::BufferType::ignore>();
}

/// @brief Pass a \p Container of \c MPI_Status to the underlying MPI call in which the statuses are stored upon
/// completion. The container may be resized according the provided \p resize_policy.
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// @tparam Container the container type to use for the statuses.
template <BufferResizePolicy resize_policy = BufferResizePolicy::no_resize, typename Container>
inline auto statuses_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::statuses,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy,
        MPI_Status>(std::forward<Container>(container));
}

/// @brief Internally construct a new \p Container of \c MPI_Status, which will hold the returned statuses.
template <typename Container>
inline auto statuses_out(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::statuses,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit,
        MPI_Status>(alloc_new<Container>);
}

/// @brief Internally construct a new \p Container<MPI_Status> which will hold the returned statuses.
template <template <typename...> typename Container>
inline auto statuses_out(AllocNewUsingT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::statuses,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit,
        MPI_Status>(alloc_new_using<Container>);
}

/// @brief Internally construct a container of \c MPI_Status, which will hold the returned statuses. The container's
/// type is usually determined by operations called on a \ref RequestPool, and defaults to \ref
/// RequestPool::default_container_type.
inline auto statuses_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::statuses,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit,
        MPI_Status>(alloc_container_of<MPI_Status>);
}

/// @}
} // namespace params
using namespace params;
} // namespace kamping
