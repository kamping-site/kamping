// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.
/// @file
/// @brief Factory methods for buffer wrappers

#pragma once

#include "buffers.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

///@brief Generates buffer wrapper based on a container for the send buffer, i.e. the underlying storage must contain
/// the data elements to send.
///
/// The underlying container must provide a \c data() member function and expose the contained \c value_type
///@tparam Container Container type which contains the elements to send.
///@param container Container which contains the elements to send
///@return ContainerBasedConstBuffer refering to the storage containing the data elements to send.
template <typename Container>
internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_buf> send_buf(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_buf>(container);
}

///@brief Generates buffer wrapper based on a container for the send counts, i.e. the underlying storage must contain
/// the send counts to each relevant PE.
///
/// The underlying container must provide a \c data() member function and expose the contained \c value_type
///@tparam Container Container type which contains the send counts.
///@param container Container which contains the send counts.
///@return ContainerBasedConstBuffer refering to the storage containing the send counts.
template <typename Container>
internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_counts>
send_counts(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_counts>(container);
}

///@brief Generates buffer wrapper based on a container for the recv counts, i.e. the underlying storage must contain
/// the recv counts from each relevant PE.
///
/// The underlying container must provide a \c data() member function and expose the contained \c value_type
///@tparam Container Container type which contains the recv counts.
///@param container Container which contains the recv counts.
///@return ContainerBasedConstBuffer refering to the storage containing the recv counts.
template <typename Container>
internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_counts>
recv_counts_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_counts>(container);
}

///@brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage must
/// contain the send displacements to each relevant PE.
///
/// The underlying container must provide a \c data() member function and expose the contained \c value_type
///@tparam Container Container type which contains the send displacements.
///@param container Container which contains the send displacements.
///@return ContainerBasedConstBuffer refering to the storage containing the send displacements.
template <typename Container>
internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_displs>
send_displs_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_displs>(container);
}

///@brief Generates buffer wrapper based on a container for the recv displacements, i.e. the underlying storage must
/// contain the recv displacements from each relevant PE.
///
/// The underlying container must provide a \c data() member function and expose the contained \c value_type
///@tparam Container Container type which contains the recv displacements.
///@param container Container type which contains the recv displacements.
///@return ContainerBasedConstBuffer refering to the storage containing the recv displacements.
template <typename Container>
internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_displs>
recv_displs_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_displs>(container);
}

// TODO adjust docu
///@brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contained the received elements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the recv displacements.
///@param container Container which will contain the received elements.
///@return UserAllocatedContainerBasedBuffer refering to the storage containing the received elements.
template <typename Container>
internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>
recv_buf(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>(container);
}

template <typename Container>
internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>
recv_buf(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>();
}

template <typename Container>
internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>
send_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>(container);
}

template <typename Container>
internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>
send_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>();
}

template <typename Container>
internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>
recv_counts_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>(container);
}

template <typename Container>
internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>
recv_counts_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>();
}

template <typename Container>
internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>
recv_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>(container);
}

template <typename Container>
internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>
recv_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>();
}

/// @}
} // namespace kamping
