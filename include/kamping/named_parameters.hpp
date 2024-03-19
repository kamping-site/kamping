// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
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

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "kamping/data_buffer.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/operation_builder.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/request.hpp"
#include "kamping/serialization.hpp"

namespace kamping {

namespace internal {
/// @brief An unused template parameter
struct unused_tparam {};
} // namespace internal

/// @addtogroup kamping_mpi_utility
/// @{

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
    return internal::
        make_empty_data_buffer_builder<Data, internal::ParameterType::send_buf, internal::BufferType::ignore>();
}

/// @brief Passes a container/single value as a send buffer to the underlying MPI call.
///
/// If data provides \c data(), it is assumed that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single value is assumed
/// @tparam Data Data type representing the element(s) to send.
/// @param data Data (either a container which contains the elements or the element directly) to send
/// @return Parameter object referring to the storage containing the data elements to send.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Data, typename Enable = std::enable_if_t<!internal::is_serialization_buffer_v<Data>>>
auto send_buf(Data&& data) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::forward<Data>(data));
}

/// @brief Passes a container/single value as a send buffer to the underlying MPI call. Additionally indicates to use
/// serialization to transfer the data.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <
    typename SerializationBufferType,
    std::enable_if_t<internal::is_serialization_buffer_v<SerializationBufferType>, bool> = true>
auto send_buf(SerializationBufferType&& data) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::forward<SerializationBufferType>(data));
}

/// @brief Passes the data provided as an initializer list as a send buffer to the underlying MPI call.
///
/// @tparam T The type of the elements in the initializer list.
/// @param data An initializer list of the data elements.
/// @return Parameter object referring to the storage containing the data elements to send.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
auto send_buf(std::initializer_list<T> data) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::move(data));
}

/// @brief Passes a container/single value as a send or receive buffer to the underlying MPI call.
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying container shall be resized. If
/// omitted, the resize policy is BufferResizePolicy::no_resize, indicating that the container should not be resized by
/// kamping.
/// @tparam Data Data type representing the element(s) to send/receive.
/// @param data Data (either a container which contains the elements or the element directly) to send or the buffer to
/// receive into.
/// @return Parameter object referring to the storage containing the data elements to send or receive.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <
    BufferResizePolicy resize_policy = BufferResizePolicy::no_resize,
    typename Data,
    typename Enable = std::enable_if_t<!internal::is_serialization_buffer_v<Data>>>
auto send_recv_buf(Data&& data) {
    constexpr internal::BufferModifiability modifiability = std::is_const_v<std::remove_reference_t<Data>>
                                                                ? internal::BufferModifiability::constant
                                                                : internal::BufferModifiability::modifiable;
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_buf,
        modifiability,
        internal::BufferType::in_out_buffer,
        resize_policy>(std::forward<Data>(data));
}

/// @brief Passes a container/single value as a send or receive buffer to the underlying MPI call. Additionally
/// indicates to use serialization to transfer the data.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <
    typename SerializationBufferType,
    typename Enable = std::enable_if_t<internal::is_serialization_buffer_v<SerializationBufferType>>>
auto send_recv_buf(SerializationBufferType&& buffer) {
    constexpr internal::BufferModifiability modifiability =
        std::is_const_v<std::remove_reference_t<SerializationBufferType>> ? internal::BufferModifiability::constant
                                                                          : internal::BufferModifiability::modifiable;
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_buf,
        modifiability,
        internal::BufferType::in_out_buffer,
        BufferResizePolicy::resize_to_fit>(std::forward<SerializationBufferType>(buffer));
}

/// @brief Indicates to use an object of type \tparam Container as `send_recv_buf`.
/// Container must provide \c data(), \c size(), \c resize(unsigned int) member functions and expose the contained \c
/// value_type.
/// @return Parameter object referring to the storage containing the data elements to send or receive.
template <typename Container>
auto send_recv_buf(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::in_out_buffer,
        BufferResizePolicy::resize_to_fit>(alloc_new<Container>);
}

/// @brief Indicates to use a parameter object encapsulating an underlying container with a \c value_type \tparam
/// ValueType as `send_recv_buf`. The type of the underlying container is determined by the MPI operation and usually
/// defaults to \ref Communicator::default_container_type.
/// @tparam ValueType The type of the elements in the buffer.
template <typename ValueType>
auto send_recv_buf(AllocContainerOfT<ValueType>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::in_out_buffer,
        BufferResizePolicy::resize_to_fit>(alloc_container_of<ValueType>);
}

/// @brief Passes a container as send counts to the underlying call, i.e. the container's storage must
/// contain the send count to each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type.
/// @tparam Container Container type which contains the send counts.
/// @param container Container which contains the send counts.
/// @return Parameter object referring to the storage containing the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto send_counts(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::forward<Container>(container));
}

/// @brief Passes the initializer list as send counts to the underlying call.
///
/// @tparam Type The type of the initializer list.
/// @param counts The send counts.
/// @return Parameter object referring to the storage containing the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
auto send_counts(std::initializer_list<T> counts) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(counts));
}

/// @brief Passes a \p container, into which the send counts deduced by KaMPIng will be written, to the underlying call.
/// \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The send counts container will be returned as part of the underlying call's result object if it is moved/passed by
/// value (e.g. `send_counts_out(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain  the send counts.
/// @param container Container which will contain the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <BufferResizePolicy resize_policy = BufferResizePolicy::no_resize, typename Container>
auto send_counts_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy,
        int>(std::forward<Container>(container));
}

/// @brief Indicates to construct an object of type \p Container, into which the send counts deduced by KaMPIng will be
/// written, in the underlying call.
/// \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type (which must be int).
///
/// The send counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send counts.
/// @return Parameter object referring to the storage which will contain the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto send_counts_out(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new<Container>);
}

/// @brief Indicates to construct a container with type \p Container<int>, into which the send counts deduced by KaMPIng
/// will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The send counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send counts.
/// @return Parameter object referring to the storage which will contain the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <template <typename...> typename Container>
auto send_counts_out(AllocNewUsingT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new_using<Container>);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<int>, into
/// which the send counts deduced by KaMPIng will be written, in the underlying call.
///
/// The send counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send counts.
/// @return Parameter object referring to the storage which will contain the send counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_counts_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<int>);
}

/// @brief Passes \p count as send count to the underlying call.
/// @param count The send count.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_count(int count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_count,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(count));
}

/// @brief Passes \p count, into which the send count deduced by KaMPIng will be written, to the underlying call.
/// @param count Reference to the location at which the send count will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_count_out(int& count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(count);
}

/// @brief Indicates to deduce the send count and return it to the caller as part of the underlying call's result
/// object.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_count_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(alloc_new<int>);
}

/// @brief Passes a container as recv counts to the underlying call, i.e. the container's storage must
/// contain the recv count from each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type.
/// @tparam Container Container type which contains the send counts.
/// @param container Container which contains the recv counts.
/// @return Parameter object referring to the storage containing the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto recv_counts(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::forward<Container>(container));
}

/// @brief Passes the initializer list as recv counts to the underlying call.
///
/// @tparam Type The type of the initializer list.
/// @param counts The recv counts.
/// @return Parameter object referring to the storage containing the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
auto recv_counts(std::initializer_list<T> counts) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(counts));
}

/// @brief Indicates that the recv counts are ignored.
inline auto recv_counts(internal::ignore_t<void> ignore [[maybe_unused]]) {
    return internal::
        make_empty_data_buffer_builder<int, internal::ParameterType::recv_counts, internal::BufferType::ignore>();
}

/// @brief Passes a \p container, into which the recv counts deduced by KaMPIng will be written, to the underlying call.
/// \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The recv counts container will be returned as part of the underlying call's result object if it is moved/passed by
/// value (e.g. `recv_counts_out(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain the recv counts.
/// @param container Container which will contain the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <BufferResizePolicy resize_policy = BufferResizePolicy::no_resize, typename Container>
auto recv_counts_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy,
        int>(std::forward<Container>(container));
}

/// @brief Indicates to construct an object of type \p Container, into which the recv counts deduced by KaMPIng will be
/// written, in the underlying call.
/// \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type (which must be int).
///
/// The recv counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the recv counts.
/// @return Parameter object referring to the storage which will contain the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Data>
auto recv_counts_out(AllocNewT<Data> container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(container);
}

/// @brief Indicates to construct a container with type \p Container<int>, into which the recv counts deduced by KaMPIng
/// will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The recv counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the recv counts.
/// @return Parameter object referring to the storage which will contain the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <template <typename...> typename Data>
auto recv_counts_out(AllocNewUsingT<Data> container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(container);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<int>, into
/// which the recv counts deduced by KaMPIng will be written, in the underlying call.
///
/// The recv counts container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the recv counts.
/// @return Parameter object referring to the storage which will contain the recv counts.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_counts_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_counts,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<int>);
}

/// @brief Passes \p count as recv count to the underlying call.
/// @param count The recv count.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_count(int count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_count,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(count));
}

/// @brief Passes \p count, into which the recv count deduced by KaMPIng will be written, to the underlying call.
/// @param count Reference to the location at which the recv count will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_count_out(int& count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(count);
}

/// @brief Indicates to deduce the recv count and return it to the caller as part of the underlying call's result
/// object.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_count_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(alloc_new<int>);
}

/// @brief Passes \p count as send/recv count to the underlying call.
/// @param count The send/recv count.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_count(int count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_count,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(count));
}

/// @brief Passes \p count, into which the send/recv count deduced by KaMPIng will be written, to the underlying call.
/// @param count Reference to the location at which the send/recv count will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_count_out(int& count) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(count);
}

/// @brief Indicates to deduce the send/recv count and return it to the caller as part of the underlying call's result
/// object.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_count_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_count,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int>(alloc_new<int>);
}

/// @brief Passes a container as send displacements to the underlying call, i.e. the container's storage must
/// contain the send displacement for each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type.
/// @tparam Container Container type which contains the send displacements.
/// @param container Container which contains the send displacements.
/// @return Parameter object referring to the storage containing the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto send_displs(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::forward<Container>(container));
}

/// @brief Passes the initializer list as send displacements to the underlying call.
///
/// @tparam Type The type of the initializer list.
/// @param displs The send displacements.
/// @return Parameter object referring to the storage containing the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
auto send_displs(std::initializer_list<T> displs) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(displs));
}

/// @brief Passes a \p container, into which the send displacements deduced by KaMPIng will be written, to the
/// underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The send displacements container will be returned as part of the underlying call's result object if it is
/// moved/passed by value (e.g. `send_counts_out(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain  the send displacements.
/// @param container Container which will contain the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <BufferResizePolicy resize_policy = BufferResizePolicy::no_resize, typename Container>
auto send_displs_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy,
        int>(std::forward<Container>(container));
}

/// @brief Indicates to construct an object of type \p Container, into which the send displacements deduced by KaMPIng
/// will be written, in the underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type (which must be int).
///
/// The send displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send displacements.
/// @return Parameter object referring to the storage which will contain the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto send_displs_out(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new<Container>);
}

/// @brief Indicates to construct a container with type \p Container<int>, into which the send displacements deduced by
/// KaMPIng will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The send displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send displacements.
/// @return Parameter object referring to the storage which will contain the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <template <typename...> typename Container>
auto send_displs_out(AllocNewUsingT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new_using<Container>);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<int>, into
/// which the send displacements deduced by KaMPIng will be written, in the underlying call.
///
/// The send displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send displacements.
/// @return Parameter object referring to the storage which will contain the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_displs_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<int>);
}

/// @brief Passes a container as receive displacements to the underlying call, i.e. the container's storage must
/// contain the receive displacement for each relevant PE.
///
/// The underlying container must provide \c data() and \c size() member functions and expose the contained \c
/// value_type.
/// @tparam Container Container type which contains the receive displacements.
/// @param container Container which contains the receive displacements.
/// @return Parameter object referring to the storage containing the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto recv_displs(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::forward<Container>(container));
}

/// @brief Passes the initializer list as receive displacements to the underlying call.
///
/// @tparam Type The type of the initializer list.
/// @param displs The receive displacements.
/// @return Parameter object referring to the storage containing the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
auto recv_displs(std::initializer_list<T> displs) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        int>(std::move(displs));
}

/// @brief Passes a \p container, into which the receive displacements deduced by KaMPIng will be written, to the
/// underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The receive displacements container will be returned as part of the underlying call's result object if it is
/// moved/passed by value (e.g. `receive_counts_out(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain  the receive displacements.
/// @param container Container which will contain the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <BufferResizePolicy resize_policy = BufferResizePolicy::no_resize, typename Container>
auto recv_displs_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy,
        int>(std::forward<Container>(container));
}

/// @brief Indicates to construct an object of type \p Container, into which the receive displacements deduced by
/// KaMPIng will be written, in the underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type (which must be int).
///
/// The receive displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the receive displacements.
/// @return Parameter object referring to the storage which will contain the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto recv_displs_out(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new<Container>);
}

/// @brief Indicates to construct a container with type \p Container<int>, into which the receive displacements deduced
/// by KaMPIng will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The receive displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the receive displacements.
/// @return Parameter object referring to the storage which will contain the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <template <typename...> typename Container>
auto recv_displs_out(AllocNewUsingT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int>(alloc_new_using<Container>);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<int>, into
/// which the receive displacements deduced by KaMPIng will be written, in the underlying call.
///
/// The receive displacements container will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the receive displacements.
/// @return Parameter object referring to the storage which will contain the receive displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_displs_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_displs,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<int>);
}

/// @brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contain the received elements when the \c MPI call has been completed.
/// The underlying container must provide \c data() and
/// \c size() member functions and expose the contained \c value_type. If a resize policy other than
/// BufferResizePolicy::do_not_resize is selected, the container must also provide a \c resize() member function.
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized.
/// @tparam Container Container type which contains the received elements.
/// @param container Container which will contain the received elements.
/// @return Object referring to the storage containing the received elements.
template <
    BufferResizePolicy resize_policy = BufferResizePolicy::no_resize,
    typename Container,
    typename Enable = std::enable_if_t<!internal::is_serialization_buffer_v<Container>>>
auto recv_buf_out(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_policy>(std::forward<Container>(container));
}

/// @brief Generates buffer wrapper based on a container for the receive buffer, i.e. the underlying storage
/// will contain the received elements when the \c MPI call has been completed.
/// The underlying container must provide \c data() and
/// \c size() member functions and expose the contained \c value_type. If a resize policy other than
/// BufferResizePolicy::do_not_resize is selected, the container must also provide a \c resize() member function.
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized.
/// @tparam Container Container type which contains the received elements.
/// @param container Container which will contain the received elements.
/// @return Object referring to the storage containing the received elements.
/// @see Alias for \ref recv_buf_out(Container&&).
template <
    BufferResizePolicy resize_policy = BufferResizePolicy::no_resize,
    typename Container,
    typename Enable = std::enable_if_t<!internal::is_serialization_buffer_v<Container>>>
auto recv_buf(Container&& container) {
    return recv_buf_out<resize_policy>(std::forward<Container>(container));
}

/// @brief A recv buffer wrapper based on a serialization buffer. Create one by using \c kamping::as_deserialized().
template <
    typename SerializationBufferType,
    typename Enable = std::enable_if_t<internal::is_serialization_buffer_v<SerializationBufferType>>>
auto recv_buf_out(SerializationBufferType&& buffer) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit>(std::forward<SerializationBufferType>(buffer));
}

/// @brief A recv buffer wrapper based on a serialization buffer. Create one by using \c kamping::as_deserialized().
/// @see Alias for \ref recv_buf_out(SerializationBufferType&&).
template <
    typename SerializationBufferType,
    typename Enable = std::enable_if_t<internal::is_serialization_buffer_v<SerializationBufferType>>>
auto recv_buf(SerializationBufferType&& buffer) {
    return recv_buf_out(std::forward<SerializationBufferType>(buffer));
}

/// @brief Generates a buffer wrapper based on a library allocated container for the receive buffer.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements. Container must provide \c data() and \c
/// size() and \c resize() member functions and expose the contained \c value_type.
/// @return Object referring to the storage containing the received elements.
template <typename Data>
auto recv_buf_out(AllocNewT<Data>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        internal::maximum_viable_resize_policy<Data>>(alloc_new<Data>);
}

/// @brief Generates a buffer wrapper based on a library allocated container for the receive buffer.
/// The underlying container must provide a \c data(), \c resize() and \c size() member function and expose the
/// contained \c value_type
/// @tparam Container Container type which contains the send displacements. Container must provide \c data() and \c
/// size() and \c resize() member functions and expose the contained \c value_type.
/// @return Object referring to the storage containing the received elements.
/// @see Alias for \ref recv_buf_out(AllocNewT<Data>).
template <typename Data>
auto recv_buf(AllocNewT<Data> tag) {
    return recv_buf_out(tag);
}

/// @brief Construct a recv buffer using \p ValueType as the underlying value type. The kind of container is determined
/// by the MPI operation and usually defaults to \ref Communicator::default_container_type.
/// @tparam ValueType The type of the elements in the buffer.
template <typename ValueType>
auto recv_buf_out(AllocContainerOfT<ValueType>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<ValueType>);
}

/// @brief Construct a recv buffer using \p ValueType as the underlying value type. The kind of container is determined
/// by the MPI operation and usually defaults to \ref Communicator::default_container_type.
/// @tparam ValueType The type of the elements in the buffer.
/// @see Alias for \ref recv_buf_out(AllocContainerOfT<ValueType>).
template <typename ValueType>
auto recv_buf(AllocContainerOfT<ValueType> tag) {
    return recv_buf_out(tag);
}

/// @brief Generates an object encapsulating the rank of the root PE. This is useful for \c MPI functions like
/// \c MPI_Gather.
///
/// @param rank Rank of the root PE.
/// @returns Root Object containing the rank information of the root PE.
inline auto root(int rank) {
    return internal::RootDataBuffer(rank);
}

/// @brief Generates an object encapsulating the rank of the root PE. This is useful for \c MPI functions like
/// \c MPI_Gather.
///
/// @param rank Rank of the root PE.
/// @returns Root Object containing the rank information of the root PE.
inline auto root(size_t rank) {
    return root(asserting_cast<int>(rank));
}

/// @brief Generates an object encapsulating the rank of the destination PE in point
/// to point communication.
///
/// @param rank The rank.
/// @returns The destination parameter.
inline auto destination(int rank) {
    return internal::RankDataBuffer<internal::RankType::value, internal::ParameterType::destination>(rank);
}

/// @brief Generates an object encapsulating the rank of the destination PE in point to point communication.
///
/// @param rank The rank.
/// @returns The destination parameter.
inline auto destination(size_t rank) {
    return destination(asserting_cast<int>(rank));
}

/// @brief Generates an object encapsulating the dummy rank \c MPI_PROC_NULL for the destination PE in point to point
/// communication.
///
/// @returns The destination parameter.
inline auto destination(internal::rank_null_t) {
    return internal::RankDataBuffer<internal::RankType::null, internal::ParameterType::destination>{};
}

/// @brief Generates an object encapsulating the rank of the source PE in
/// point to point communication.
///
/// @param rank The rank.
/// @returns The source parameter.
inline auto source(int rank) {
    return internal::RankDataBuffer<internal::RankType::value, internal::ParameterType::source>(rank);
}

/// @brief Generates an object encapsulating the rank of the source PE in
/// point to point communication.
///
/// @param rank The rank.
/// @returns The source parameter.
inline auto source(size_t rank) {
    return source(asserting_cast<int>(rank));
}

/// @brief Use an arbitrary rank as source in a point to point communication.
inline auto source(internal::rank_any_t) {
    return internal::RankDataBuffer<internal::RankType::any, internal::ParameterType::source>{};
}

/// @brief Use the dummy rank \c MPI_PROC_NULL as source in a point to point communication.
inline auto source(internal::rank_null_t) {
    return internal::RankDataBuffer<internal::RankType::null, internal::ParameterType::source>{};
}

/// @brief Use an arbitrary message tag for \c kamping::Communicator::probe() or \c kamping::Communicator::recv().
inline auto tag(internal::any_tag_t) {
    return internal::TagParam<internal::TagType::any>{};
}

/// @brief Generates a parameter object encapsulating a tag.
/// @param value the tag value.
/// @returns The tag wrapper.
inline auto tag(int value) {
    return internal::TagParam<internal::TagType::value>{value};
}

/// @brief Generates a parameter object encapsulating a tag from an enum type.
/// The underlying type of the enum must be convertible to \c int.
/// @tparam EnumType type of the tag enum.
/// @param value the tag value.
/// @returns The tag wrapper.
template <typename EnumType, typename = std::enable_if_t<std::is_enum_v<EnumType>>>
inline auto tag(EnumType value) {
    static_assert(
        std::is_convertible_v<std::underlying_type_t<EnumType>, int>,
        "The underlying enum type must be implicitly convertible to int."
    );
    return tag(static_cast<int>(value));
}

/// @brief Pass a request handle to the underlying MPI call.
/// @param request The request handle.
inline auto request(Request& request) {
    return internal::make_data_buffer<
        internal::ParameterType,
        internal::ParameterType::request,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize>(request);
}

/// @brief Pass a request from a \ref RequestPool to the underlying MPI call.
/// @param request The request handle.
/// @tparam IndexType The type of the index used by the \ref RequestPool for requests.
template <typename IndexType>
inline auto request(PooledRequest<IndexType> request) {
    return internal::make_data_buffer<
        internal::ParameterType,
        internal::ParameterType::request,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize>(std::move(request));
}

/// @brief Internally allocate a request object and return it to the user.
inline auto request() {
    return internal::make_data_buffer<
        internal::ParameterType,
        internal::ParameterType::request,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize>(alloc_new<Request>);
}

/// @brief Send mode parameter for point to point communication.
/// Pass any of the tags from the \c kamping::send_modes namespace.
template <typename SendModeTag>
inline auto send_mode(SendModeTag) {
    return internal::SendModeParameter<SendModeTag>{};
}

/// @brief generates a parameter object for a reduce operation. Accepts function objects, lambdas, function pointers or
/// native \c MPI_Op as argument.
///
/// @tparam Op the type of the operation
/// @tparam Commutative tag whether the operation is commutative
/// @param op the operation
/// @param commute the commutativity tag
///     May be any instance of \c commutative, \c or non_commutative. Passing \c undefined_commutative is only
///     supported for builtin and native operations. This is used to streamline the interface so that the use does not
///     have to provide commutativity info when the operation is builtin.
template <typename Op, typename Commutative = ops::internal::undefined_commutative_tag>
internal::OperationBuilder<Op, Commutative>
op(Op&& op, Commutative commute = ops::internal::undefined_commutative_tag{}) {
    return internal::OperationBuilder<Op, Commutative>(std::forward<Op>(op), commute);
}

/// @brief Generates an object encapsulating the value to return on the first rank in \c exscan().
///
/// @param container Value(s) to return on the first rank.
/// @returns OnRank0 Object containing the information which value to return on the first rank.
template <typename Container>
inline auto values_on_rank_0(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::values_on_rank_0,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::forward<Container>(container));
}

/// @brief Generates an object encapsulating the value to return on the first rank in \c exscan().
///
/// @param values Value(s) to return on the first rank.
/// @returns OnRank0 Object containing the information which value to return on the first rank.
// TODO zero-overhead
template <typename T>
inline auto values_on_rank_0(std::initializer_list<T> values) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::values_on_rank_0,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::move(values));
}

/// @brief The send type to use in the respective \c MPI operation.
/// @param send_type MPI_Datatype to use in the wrapped \c MPI operation.
/// @return The corresponding parameter object.
inline auto send_type(MPI_Datatype send_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(send_type));
}

/// @brief Output parameter for the send type.
/// @return The corresponding parameter object.
inline auto send_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Output parameter for the send type.
/// The type will be stored at the location referred to by the provided reference.
/// @param send_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
inline auto send_type_out(MPI_Datatype& send_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(send_type);
}

/// @brief The recv type to use in the respective \c MPI operation.
/// @param recv_type MPI_Datatype to use in the wrapped \c MPI operation.
/// @return The corresponding parameter object.
inline auto recv_type(MPI_Datatype recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(recv_type));
}

/// @brief Output parameter for the recv type.
/// @return The corresponding parameter object.
inline auto recv_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Output parameter for the recv type.
/// The type will be stored at the location referred to by the provided reference.
/// @param recv_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
inline auto recv_type_out(MPI_Datatype& recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(recv_type);
}

/// @brief The send_recv type to use in the respective \c MPI operation.
/// @param send_recv_type MPI_Datatype to use in the wrapped \c MPI operation.
/// @return The corresponding parameter object.
inline auto send_recv_type(MPI_Datatype send_recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(send_recv_type));
}

/// @brief Output parameter for the send_recv type.
/// @return The corresponding parameter object.
inline auto send_recv_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Output parameter for the send_recv type.
/// The type will be stored at the location referred to by the provided reference.
/// @param send_recv_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
inline auto send_recv_type_out(MPI_Datatype& send_recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(send_recv_type);
}
/// @}
} // namespace kamping
