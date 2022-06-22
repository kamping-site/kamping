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
#include <utility>

#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {
/// @brief Tag type for parameters that can be omitted on some PEs (e.g., root PE, or non-root PEs).
template <typename T>
struct ignore_t {};

/// @brief Creates a user allocated DataBuffer containing the supplied data (a container or a single element)
///
/// Creates a user allocated DataBuffer with the given template parameters and ownership based on whether an rvalue or
/// lvalue reference is passed.
///
/// @tparam parameter_type parameter type represented by this buffer.
/// @tparam modifiability `modifiable` if a KaMPIng operation is allowed to
/// modify the underlying container. `constant` otherwise.
/// @tparam Data Container or data type on which this buffer is based.
/// @param data Universal reference to a container or single element holding the data for the buffer.
///
/// @return A user allocated DataBuffer with the given template parameters and matching ownership.
template <ParameterType parameter_type, BufferModifiability modifiability, typename Data>
auto make_data_buffer(Data&& data) {
    constexpr BufferOwnership ownership =
        std::is_rvalue_reference_v<Data&&> ? BufferOwnership::owning : BufferOwnership::referencing;

    // Make sure that Data is const, the buffer created is constant (so we don't really remove constness in the return
    // statement below).
    constexpr bool is_const_data_type = std::is_const_v<std::remove_reference_t<Data>>;
    constexpr bool is_const_buffer    = modifiability == BufferModifiability::constant;
    // Implication: is_const_data_type => is_const_buffer.
    static_assert(!is_const_data_type || is_const_buffer);

    return DataBuffer<
        std::remove_const_t<std::remove_reference_t<Data>>, parameter_type, modifiability, ownership,
        BufferAllocation::user_allocated>(std::forward<Data>(data));
}

/// @brief Creates a library allocated DataBuffer containing the supplied data (a container or a single element)
///
/// Creates a library allocated DataBuffer with the given template parameters.
///
/// @tparam parameter_type parameter type represented by this buffer.
/// @tparam modifiability `modifiable` if a KaMPIng operation is allowed to
/// modify the underlying container. `constant` otherwise.
/// @tparam Data Container or data type on which this buffer is based.
///
/// @return A library allocated DataBuffer with the given template parameters.
template <ParameterType parameter_type, BufferModifiability modifiability, typename Data>
auto make_data_buffer(NewContainer<Data>&&) {
    return DataBuffer<Data, parameter_type, modifiability, BufferOwnership::owning, BufferAllocation::lib_allocated>();
}
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
auto send_buf(Data&& data) {
    return internal::make_data_buffer<internal::ParameterType::send_buf, internal::BufferModifiability::constant, Data>(
        std::forward<Data>(data));
}

/// @brief Generates a buffer taking ownership of the data pass to the send buffer as an initializer list.
///
/// @tparam T The type of the elements in the initializer list.
/// @param data An initializer list of the data elements.
/// @return Object referring to the storage containing the data elements to send.
template <typename T>
auto send_buf(std::initializer_list<T> data) {
    std::vector<T> data_vec{data};
    return internal::make_data_buffer<
        internal::ParameterType::send_buf, internal::BufferModifiability::constant, std::vector<T>>(
        std::move(data_vec));
}

/// @brief Generates a buffer wrapper encapsulating a buffer used for sending or receiving based on this processes rank
/// and the root() of the operation. This buffer type may encapsulate const data and in which case it can only be used
/// as the send buffer. For some functions (e.g. bcast), you have to pass a send_recv_buf as the send buffer.
///
/// @tparam Data Data type representing the element(s) to send/receive.
/// @param data Data (either a container which contains the elements or the element directly) to send or the buffer to
/// receive into.
/// @return Object referring to the storage containing the data elements to send / the received elements.
template <typename Data>
auto send_recv_buf(Data&& data) {
    if constexpr (std::is_const_v<std::remove_reference_t<Data>>) {
        return internal::make_data_buffer<
            internal::ParameterType::send_recv_buf, internal::BufferModifiability::constant, Data>(
            std::forward<Data>(data));
    } else {
        return internal::make_data_buffer<
            internal::ParameterType::send_recv_buf, internal::BufferModifiability::modifiable, Data>(
            std::forward<Data>(data));
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
    return internal::make_data_buffer<
        internal::ParameterType::send_recv_buf, internal::BufferModifiability::modifiable, Container>(
        NewContainer<Container>{});
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
auto send_counts(Container&& container) {
    return internal::make_data_buffer<
        internal::ParameterType::send_counts, internal::BufferModifiability::constant, Container>(
        std::forward<Container>(container));
}

/// @brief Generates a buffer wrapper for the send counts based on an initializer list, i.e. the
/// send counts to each relevant PE.
///
/// @tparam Type The type of the initializer list.
/// @param counts The send counts.
/// @return Object referring to the storage containing the send counts.
template <typename T>
auto send_counts(std::initializer_list<T> counts) {
    std::vector<T> counts_vec{counts};

    return internal::make_data_buffer<
        internal::ParameterType::send_counts, internal::BufferModifiability::constant, std::vector<T>>(
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
auto recv_counts(Container&& container) {
    return internal::make_data_buffer<
        internal::ParameterType::recv_counts, internal::BufferModifiability::constant, Container>(
        std::forward<Container>(container));
}

/// @brief Generates a buffer wrapper for the recv counts based on an initializer list, i.e. the
/// recv counts from each relevant PE.
///
/// @tparam Type The type of the initializer list.
/// @param counts The recv counts.
/// @return Object referring to the storage containing the recv counts.
template <typename T>
auto recv_counts(std::initializer_list<T> counts) {
    std::vector<T> counts_vec{counts};

    return internal::make_data_buffer<
        internal::ParameterType::recv_counts, internal::BufferModifiability::constant, std::vector<T>>(
        std::move(counts_vec));
}

/// @brief Generates a wrapper for a recv count input parameter.
/// @param recv_count The recv count to be encapsulated.
/// @return Wrapper around the given recv count.
inline auto recv_count(int recv_count) {
    return internal::make_data_buffer<internal::ParameterType::recv_count, internal::BufferModifiability::constant>(
        std::move(recv_count));
    // return internal::SingleElementOwningBuffer<int, internal::ParameterType::recv_count>(recv_count);
}

/// @brief Generates a wrapper for a recv count output parameter.
/// @param recv_count_out Reference for the output parameter.
/// @return Wrapper around the given reference.
inline auto recv_count_out(int& recv_count_out) {
    return internal::make_data_buffer<internal::ParameterType::recv_count, internal::BufferModifiability::modifiable>(
        recv_count_out);
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
auto send_displs(Container&& container) {
    return internal::make_data_buffer<internal::ParameterType::send_displs, internal::BufferModifiability::constant>(
        std::forward<Container>(container));
}

/// @brief Generates a buffer wrapper for the send displacements based on an initializer list, i.e. the
/// send displacements from each relevant PE.
///
/// @tparam Type The type of the initializer list.
/// @param displs The send displacements.
/// @return Object referring to the storage containing the send displacements.
template <typename T>
auto send_displs(std::initializer_list<T> displs) {
    std::vector<T> displs_vec{displs};
    return internal::make_data_buffer<internal::ParameterType::send_displs, internal::BufferModifiability::constant>(
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
auto recv_displs(Container&& container) {
    return internal::make_data_buffer<internal::ParameterType::recv_displs, internal::BufferModifiability::constant>(
        std::forward<Container>(container));
}

/// @brief Generates a buffer wrapper for the receive displacements based on an initializer list, i.e. the
/// receive displacements from each relevant PE.
///
/// @tparam Type The type of the initializer list.
/// @param displs The receive displacements.
/// @return Object referring to the storage containing the receive displacements.
template <typename T>
auto recv_displs(std::initializer_list<T> displs) {
    std::vector<T> displs_vec{displs};
    return internal::make_data_buffer<internal::ParameterType::recv_displs, internal::BufferModifiability::constant>(
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
auto recv_buf(Container&& container) {
    return internal::make_data_buffer<internal::ParameterType::recv_buf, internal::BufferModifiability::modifiable>(
        std::forward<Container>(container));
}

/// @brief Generates buffer wrapper based on a container for the send displacements, i.e. the underlying storage
/// will contained the send displacements when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements.
/// @param container Container which will contain the send displacements.
/// @return Object referring to the storage containing the send displacements.
template <typename Container>
auto send_displs_out(Container&& container) {
    return internal::make_data_buffer<internal::ParameterType::send_displs, internal::BufferModifiability::modifiable>(
        std::forward<Container>(container));
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
    return internal::make_data_buffer<internal::ParameterType::send_displs, internal::BufferModifiability::modifiable>(
        NewContainer<Container>{});
}

/// @brief Generates buffer wrapper based on a container for the receive counts, i.e. the underlying storage
/// will contained the receive counts when the \c MPI call has been completed.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the receive counts.
/// @param container Container which will contain the receive counts.
/// @return Object referring to the storage containing the receive counts.
template <typename Container>
auto recv_counts_out(Container&& container) {
    return internal::make_data_buffer<internal::ParameterType::recv_counts, internal::BufferModifiability::modifiable>(
        std::forward<Container>(container));
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
    return internal::make_data_buffer<internal::ParameterType::recv_counts, internal::BufferModifiability::modifiable>(
        NewContainer<Container>{});
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
