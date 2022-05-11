// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
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
/// @brief Factory methods for buffer wrappers

#pragma once

#include <initializer_list>
#include <type_traits>

#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {
/// @brief Boolean value helping to decide if data type has \c .data() method.
/// @return \c true if class has \c .data() method and \c false otherwise.
template <typename, typename = void>
constexpr bool has_data_member_v = false;

/// @brief Boolean value helping to decide if data type has \c .data() method.
/// @return \c true if class has \c .data() method and \c false otherwise.
template <typename T>
constexpr bool has_data_member_v<T, std::void_t<decltype(std::declval<T>().data())>> = true;

/// @brief Tag type for parameters that can be omitted on some PEs (e.g., root PE, or non-root PEs).
template <typename T>
struct ignore_t {};
} // namespace internal

/// @brief Tag for parameters that can be omitted on some PEs (e.g., root PE, or non-root PEs).
template <typename T>
constexpr internal::ignore_t<T> ignore{};

/// @brief Generates a dummy send buf that wraps a \c nullptr.
///
/// This is useful for operations where a send_buf is required on some PEs, such as the root PE,
/// but not all PEs that participate in the collective communication.
///
/// @tparam Data Data type for elements in the send buffer. This must be the same type as in the actual send_buf.
/// @param ignore Tag parameter for overload dispatching, pass in `kamping::ignore<Data>`.
/// @return Object wrapping a \c nullptr as a send buffer.
template <typename Data>
auto send_buf(internal::ignore_t<Data> ignore [[maybe_unused]]) {
    return internal::EmptyBuffer<Data, internal::ParameterType::send_buf>();
}

/// @brief Generates buffer wrapper based on the data in the send buffer, i.e. the underlying storage must contain
/// the data element(s) to send.
///
/// If the underlying container provides \c data(), it is assumed that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single element is wrapped in the
/// send buffer.
/// @tparam Data Data type representing the element(s) to send.
/// @param data Data (either a container which contains the elements or the element directly) to send
/// @return Object referring to the storage containing the data elements to send.
template <typename Data>
auto send_buf(const Data& data) {
    if constexpr (internal::has_data_member_v<Data>) {
        return internal::ContainerBasedConstBuffer<Data, internal::ParameterType::send_buf>(data);
    } else {
        return internal::SingleElementConstBuffer<Data, internal::ParameterType::send_buf>(data);
    }
}
template <typename T>
auto send_buf(std::initializer_list<T> data) {
    std::vector<T> data_vec{data};
    return internal::ContainerBasedOwningBuffer<std::vector<T>, internal::ParameterType::send_buf>(std::move(data_vec));
}

template <class Data, typename = std::enable_if_t<std::is_rvalue_reference<Data&&>::value>>
auto send_buf(Data&& data) {
    if constexpr (internal::has_data_member_v<Data>) {
        return internal::ContainerBasedOwningBuffer<Data, internal::ParameterType::send_buf>(std::forward<Data>(data));
    } else {
        return internal::SingleElementOwningBuffer<Data, internal::ParameterType::send_buf>(std::forward<Data>(data));
    }
}

/// @brief Generates a buffer wrapper encapsulating a buffer used for sending or receiving based on this processes rank
/// and the root() of the operation.
///
/// For example when used as parameter to \c bcast, all processes provide this buffer; on the root process it
/// acts as the send buffer, on all other processes as the receive buffer.
///
/// If the underlying container provides \c data(), it is assumed that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single element is wrapped in the
/// send_recv buffer. For receiving, the buffer is automatically resized to the correct size and thus has to provide a
/// \c resize() method.
///
/// @tparam Data Data type representing the element(s) to send/receive.
/// @param data Data (either a container which contains the elements or the element directly) to send or the buffer to
/// receive into.
/// @return Object referring to the storage containing the data elements to send / the received elements.
template <typename Data>
auto send_recv_buf(Data& data) {
    if constexpr (internal::has_data_member_v<Data>) {
        return internal::UserAllocatedContainerBasedBuffer<Data, internal::ParameterType::send_recv_buf>(data);
    } else {
        return internal::SingleElementModifiableBuffer<Data, internal::ParameterType::send_recv_buf>(data);
    }
}

/// @brief Generates a buffer wrapper encapsulating a buffer used for sending based on this processes rank and the
/// root() of the operation. This buffer type encapsulates const data and can therefore only be used as the send buffer.
/// For some functions (e.g. bcast), you have to pass a send_recv_buf as the send buffer.
///
/// If the underlying container provides \c data(), we assume that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single element is wrapped in the
/// send_recv buffer. Receiving into a constant container is not possible.
///
/// @tparam Data Data type representing the element(s) to send/receive.
/// @param data Data (either a container which contains the elements or the element directly) to send
/// @return Object referring to the storage containing the data elements to send.
template <typename Data>
auto send_recv_buf(const Data& data) {
    if constexpr (internal::has_data_member_v<Data>) {
        return internal::ContainerBasedConstBuffer<Data, internal::ParameterType::send_recv_buf>(data);
    } else {
        return internal::SingleElementConstBuffer<Data, internal::ParameterType::send_recv_buf>(data);
    }
}

/// @brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contain the received elements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the received elements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_recv_buf(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_recv_buf>();
}

/// @brief Generates buffer wrapper based on a container for the send counts, i.e. the underlying storage must contain
/// the send counts to each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
/// @tparam Container Container type which contains the send counts.
/// @param container Container which contains the send counts.
/// @return Object referring to the storage containing the send counts.
template <typename Container>
auto send_counts(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_counts>(container);
}

template <class Container, typename = std::enable_if_t<std::is_rvalue_reference<Container&&>::value>>
auto send_counts(Container&& container) {
    return internal::ContainerBasedOwningBuffer<Container, internal::ParameterType::send_counts>(
        std::forward<Container>(container));
}

template <typename T>
auto send_counts(std::initializer_list<T> counts) {
    std::vector<T> counts_vec{counts};
    return internal::ContainerBasedOwningBuffer<std::vector<T>, internal::ParameterType::send_counts>(
        std::move(counts_vec));
}

/// @brief Generates buffer wrapper based on a container for the recv counts, i.e. the underlying storage must contain
/// the recv counts from each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
/// @tparam Container Container type which contains the recv counts.
/// @param container Container which contains the recv counts.
/// @return Object referring to the storage containing the recv counts.
template <typename Container>
auto recv_counts(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_counts>(container);
}

template <class Container, typename = std::enable_if_t<std::is_rvalue_reference<Container&&>::value>>
auto recv_counts(Container&& container) {
    return internal::ContainerBasedOwningBuffer<Container, internal::ParameterType::recv_counts>(
        std::forward<Container>(container));
}

template <typename T>
auto recv_counts(std::initializer_list<T> counts) {
    std::vector<T> counts_vec{counts};
    return internal::ContainerBasedOwningBuffer<std::vector<T>, internal::ParameterType::recv_counts>(
        std::move(counts_vec));
}

/// @brief Generates a wrapper for a recv count input parameter.
/// @param recv_count The recv count to be encapsulated.
/// @return Wrapper around the given recv count.
inline auto recv_count(int const& recv_count) {
    return internal::SingleElementConstBuffer<int, internal::ParameterType::recv_count>(recv_count);
}

/// @brief Generates a wrapper for a recv count output parameter.
/// @param recv_count_out Reference for the output parameter.
/// @return Wrapper around the given reference.
inline auto recv_count_out(int& recv_count_out) {
    return internal::SingleElementModifiableBuffer<int, internal::ParameterType::recv_count>(recv_count_out);
}

/// @brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage must
/// contain the send displacements to each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
/// @tparam Container Container type which contains the send displacements.
/// @param container Container which contains the send displacements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::send_displs>(container);
}

template <class Container, typename = std::enable_if_t<std::is_rvalue_reference<Container&&>::value>>
auto send_displs(Container&& container) {
    return internal::ContainerBasedOwningBuffer<Container, internal::ParameterType::send_displs>(
        std::forward<Container>(container));
}

template <typename T>
auto send_displs(std::initializer_list<T> displs) {
    std::vector<T> displs_vec{displs};
    return internal::ContainerBasedOwningBuffer<std::vector<T>, internal::ParameterType::send_displs>(
        std::move(displs_vec));
}

/// @brief Generates buffer wrapper based on a container for the recv displacements, i.e. the underlying storage must
/// contain the recv displacements from each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type
/// @tparam Container Container type which contains the recv displacements.
/// @param container Container type which contains the recv displacements.
/// @return Object referring to the storage containing the recv displacements.
template <typename Container>
auto recv_displs(const Container& container) {
    return internal::ContainerBasedConstBuffer<Container, internal::ParameterType::recv_displs>(container);
}

template <class Container, typename = std::enable_if_t<std::is_rvalue_reference<Container&&>::value>>
auto recv_displs(Container&& container) {
    return internal::ContainerBasedOwningBuffer<Container, internal::ParameterType::recv_displs>(
        std::forward<Container>(container));
}

template <typename T>
auto recv_displs(std::initializer_list<T> displs) {
    std::vector<T> displs_vec{displs};
    return internal::ContainerBasedOwningBuffer<std::vector<T>, internal::ParameterType::recv_displs>(
        std::move(displs_vec));
}

/// @brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contained the received elements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the received elements.
/// @param container Container which will contain the received elements.
/// @return Object referring to the storage containing the received elements.
template <typename Container>
auto recv_buf(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>(container);
}

/// @brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contained the received elements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the received elements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto recv_buf(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_buf>();
}

/// @brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage
/// will contained the send displacements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements.
/// @param container Container which will contain the send displacements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>(container);
}

/// @brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage
/// will contained the send displacements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::send_displs>();
}

/// @brief Generates buffer wrapper based on a container for the receive counts, i.e. the underlying storage
/// will contained the receive counts when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the receive counts.
/// @param container Container which will contain the receive counts.
/// @return Object referring to the storage containing the receive counts.
template <typename Container>
auto recv_counts_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>(container);
}

/// @brief Generates buffer wrapper based on a container for the receive counts, i.e. the underlying storage
/// will contained the receive counts when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements.
/// @return Object referring to the storage containing the receive counts.
template <typename Container>
auto recv_counts_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_counts>();
}

/// @brief Generates buffer wrapper based on a container for the receive displacements, i.e. the underlying storage
/// will contained the receive displacements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the receive displacements.
/// @param container Container which will contain the receive displacements.
/// @return Object referring to the storage containing the receive displacements.
template <typename Container>
auto recv_displs_out(Container& container) {
    return internal::UserAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>(container);
}

/// @brief Generates buffer wrapper based on a container for the receive displacements, i.e. the underlying storage
/// will contained the receive displacements when the \c MPI call has been completed.
/// The storage is allocated by the library and encapsulated in a container of type Container.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements.
/// @return Object referring to the storage containing the receive displacements.
template <typename Container>
auto recv_displs_out(NewContainer<Container>&&) {
    return internal::LibAllocatedContainerBasedBuffer<Container, internal::ParameterType::recv_displs>();
}

/// @brief Generates an object encapsulating the rank of the root PE. This is useful for \c MPI functions like \c
/// MPI_Gather.
///
/// @param rank Rank of the root PE.
/// @returns Root Object containing the rank information of the root PE.
inline auto root(int rank) {
    return internal::Root(rank);
}

/// @brief Generates an object encapsulating the rank of the root PE. This is useful for \c MPI functions like \c
/// MPI_Gather.
///
/// @param rank Rank of the root PE.
/// @returns Root Object containing the rank information of the root PE.
inline auto root(size_t rank) {
    return internal::Root(rank);
}

/// @brief generates a parameter object for a reduce operation.
///
/// @tparam Op the type of the operation
/// @tparam Commutative tag whether the operation is commutative
/// @param op the operation
/// @param commute the commutativity tag
///     May be any instance of \c commutative, \c or non_commutative. Passing \c undefined_commutative is only supported
///     for builtin operations. This is used to streamline the interface so that the use does not have to provide
///     commutativity info when the operation is builtin.
template <typename Op, typename Commutative = internal::undefined_commutative_tag>
internal::OperationBuilder<Op, Commutative> op(Op&& op, Commutative commute = internal::undefined_commutative_tag{}) {
    return internal::OperationBuilder<Op, Commutative>(std::forward<Op>(op), commute);
}

/// @}
} // namespace kamping
