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

#include "kamping/parameter_objects.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {
/// @brief Boolean value helping to decide if data type has \c .data() method.
/// @return \c true if class has \c .data() method and \c false otherwise.
template <typename, typename = void>
constexpr bool has_data_member_v{};

/// @brief Boolean value helping to decide if data type has \c .data() method.
/// @return \c true if class has \c .data() method and \c false otherwise.
template <typename T>
constexpr bool has_data_member_v<T, std::void_t<decltype(std::declval<T>().data())>> = true;
} // namespace internal

///@brief Generates buffer wrapper based on the data in the send buffer, i.e. the underlying storage must contain
/// the data element(s) to send.
///
/// If the underlying container provides \c data(), it is assumed that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single element is wrapped in the
/// send buffer.
///@tparam Data Data type representing the element(s) to send.
///@param data Data (either a container which contains the elements or the element directly) to send
///@return Object referring to the storage containing the data elements to send.
template <typename Data>
auto send_buf(const Data& data) {
    if constexpr (internal::has_data_member_v<Data>) {
        return internal::ContainerBasedConstBuffer<Data, internal::ParameterType::send_buf>(data);
    } else {
        return internal::SingleElementConstBuffer<Data, internal::ParameterType::send_buf>(data);
    }
}

///@brief Generates buffer wrapper based on a container for the send counts, i.e. the underlying storage must contain
/// the send counts to each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
///@tparam Container Container type which contains the send counts.
///@param container Container which contains the send counts.
///@return Object referring to the storage containing the send counts.
template <typename Container>
auto send_counts(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_counts>(container);
}

///@brief Generates buffer wrapper based on a container for the recv counts, i.e. the underlying storage must contain
/// the recv counts from each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
///@tparam Container Container type which contains the recv counts.
///@param container Container which contains the recv counts.
///@return Object referring to the storage containing the recv counts.
template <typename Container>
auto recv_counts_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_counts>(container);
}

///@brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage must
/// contain the send displacements to each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
///@tparam Container Container type which contains the send displacements.
///@param container Container which contains the send displacements.
///@return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_displs>(container);
}

///@brief Generates buffer wrapper based on a container for the recv displacements, i.e. the underlying storage must
/// contain the recv displacements from each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
///@tparam Container Container type which contains the recv displacements.
///@param container Container type which contains the recv displacements.
///@return Object referring to the storage containing the recv displacements.
template <typename Container>
auto recv_displs_in(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_displs>(container);
}

///@brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contained the received elements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the received elements.
///@param container Container which will contain the received elements.
///@return Object referring to the storage containing the received elements.
template <typename Container>
auto recv_buf(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>(container);
}

///@brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contained the received elements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the received elements.
///@return Object referring to the storage containing the send displacements.
template <typename Container>
auto recv_buf(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>();
}

///@brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage
/// will contained the send displacements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the send displacments.
///@param container Container which will contain the send displacements.
///@return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>(container);
}

///@brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage
/// will contained the send displacements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the send displacments.
///@return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>();
}

///@brief Generates buffer wrapper based on a container for the receive counts, i.e. the underlying storage
/// will contained the receive counts when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the receive counts.
///@param container Container which will contain the receive counts.
///@return Object referring to the storage containing the receive counts.
template <typename Container>
auto recv_counts_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>(container);
}

///@brief Generates buffer wrapper based on a container for the receive counts, i.e. the underlying storage
/// will contained the receive counts when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the send displacments.
///@return Object referring to the storage containing the receive counts.
template <typename Container>
auto recv_counts_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>();
}

///@brief Generates buffer wrapper based on a container for the receive displacements, i.e. the underlying storage
/// will contained the receive displacements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the receive displacements.
///@param container Container which will contain the receive displacements.
///@return Object referring to the storage containing the receive displacements.
template <typename Container>
auto recv_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>(container);
}

///@brief Generates buffer wrapper based on a container for the receive displacments, i.e. the underlying storage
/// will contained the receive displacments when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
///@tparam Container Container type which contains the send displacments.
///@return Object referring to the storage containing the receive displacments.
template <typename Container>
auto recv_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>();
}

///@brief Generates an object encapsulating the rank of the root PE. This is useful for \c MPI functions like \c
/// MPI_Gather.
///
///@param rank Rank of the root PE.
///@returns Root Object containing the rank information of the root PE.
inline auto root(int rank) {
    return internal::Root(rank);
}

/// @}
} // namespace kamping
