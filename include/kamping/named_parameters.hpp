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

namespace params {

//// @addtogroup kamping_named_parameters
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

/// @brief Passes a container/single value as rvalue as a send buffer to the underlying MPI call.
/// This transfers ownership of the data to the call and re-returns ownership to the caller as part of the result
/// object.
///
/// If data provides \c data(), it is assumed that it is a container and all elements in the
/// container are considered for the operation. In this case, the container has to provide a \c size() member functions
/// and expose the contained \c value_type. If no \c data() member function exists, a single value is assumed
/// @tparam Data Data type representing the element(s) to send.
/// @param data Data (either a container which contains the elements or the element directly) to send
/// @return Parameter object referring to the storage containing the data elements to send.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Data, typename Enable = std::enable_if_t<std::is_rvalue_reference_v<Data&&>>>
auto send_buf_out(Data&& data) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferType::in_out_buffer,
        BufferResizePolicy::no_resize>(std::forward<Data>(data));
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
/// @return Parameter object referring to the storage containing the data elements to send or receive.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
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
/// moved/passed by value (e.g. `send_displs_out(std::move(container))`).
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
/// moved/passed by value (e.g. `receive_displs_out(std::move(container))`).
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

/// @brief Passes a \p container, into which the received elements will be written, to the
/// underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The receive buffer will be returned as part of the underlying call's result object if it is
/// moved/passed by value (e.g. `recv_buf_out(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain  the send displacements.
/// @param container Container which will contain the send displacements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
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

/// @brief Passes a \p container, into which the received elements will be written, to the
/// underlying call. \p Container must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - expose \c value_type (which must be int).
/// - if \p resize_policy is not BufferResizePolicy::no_resize, \p container additionally has to expose a
/// `resize(unsigned int)` member function.
///
/// The receive buffer will be returned as part of the underlying call's result object if it is
/// moved/passed by value (e.g. `recv_buf(std::move(container))`).
///
/// @tparam resize_policy Policy specifying whether (and if so, how) the underlying buffer shall be resized. The default
/// resize policy is BufferResizePolicy::no_resize, indicating that the buffer should not be resized by KaMPIng.
/// @tparam Container Container type which will contain  the send displacements.
/// @param container Container which will contain the send displacements.
/// @see Alias for \ref recv_buf_out(Container&&).
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <
    BufferResizePolicy resize_policy = BufferResizePolicy::no_resize,
    typename Container,
    typename Enable = std::enable_if_t<!internal::is_serialization_buffer_v<Container>>>
auto recv_buf(Container&& container) {
    return recv_buf_out<resize_policy>(std::forward<Container>(container));
}

/// @brief Indicates to deserialize the received elements in the underlying call.
///
/// Example usage:
/// ```cpp
///   using dict_type = std::unordered_map<int, double>;
///   recv_buf_out(kamping::as_deserializable<dict_type>());
/// ```
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
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

/// @brief Indicates to deserialize the received elements in the underlying call.
///
/// Example usage:
/// ```cpp
///   using dict_type = std::unordered_map<int, double>;
///   recv_buf(kamping::as_deserializable<dict_type>());
/// ```
///
/// @see Alias for \ref recv_buf_out(Container&&).
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <
    typename SerializationBufferType,
    typename Enable = std::enable_if_t<internal::is_serialization_buffer_v<SerializationBufferType>>>
auto recv_buf(SerializationBufferType&& buffer) {
    return recv_buf_out(std::forward<SerializationBufferType>(buffer));
}

/// @brief Indicates to construct a container of type \p Container, into which the received elements
/// will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The receive buffer will be returned as part of the underlying call's result object
///
/// @tparam Container Container type which will contains the send displacements.
/// @return Parameter object referring to the storage which will contain the received elements.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto recv_buf_out(AllocNewT<Container>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        internal::maximum_viable_resize_policy<Container>>(alloc_new<Container>);
}

/// @brief Indicates to construct a container of type \p Container, into which the received elements
/// will be written, in the underlying call.
///
/// \p Container<int> must satisfy the following constraints:
/// - provide a \c data() member function
/// - provide a \c size() member function
/// - provide a \c resize(unsigned int) member function
/// - expose \c value_type.
///
/// The receive buffer will be returned as part of the underlying call's result object.
///
/// @tparam Container Container type which will contains the send displacements.
/// @return Parameter object referring to the storage which will contain the received elements.
/// @see Alias for \ref recv_buf_out(Container&&).
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
auto recv_buf(AllocNewT<Container> tag) {
    return recv_buf_out(tag);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<\p
/// ValueType>, into which the received elements will be written, in the underlying call.
///
/// The receive buffer will be returned as part of the underlying call's result object.
///
/// defaults to \ref Communicator::default_container_type.
/// @tparam ValueType The type of the elements in the buffer.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename ValueType>
auto recv_buf_out(AllocContainerOfT<ValueType>) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        resize_to_fit>(alloc_container_of<ValueType>);
}

/// @brief Indicates to construct a container with type \ref kamping::Communicator::default_container_type<\p
/// ValueType>, into which the received elements will be written, in the underlying call.
///
/// The receive buffer will be returned as part of the underlying call's result object.
///
/// defaults to \ref Communicator::default_container_type.
/// @tparam ValueType The type of the elements in the buffer.
/// @see Alias for \ref recv_buf_out(Container&&).
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename ValueType>
auto recv_buf(AllocContainerOfT<ValueType> tag) {
    return recv_buf_out(tag);
}

/// @brief Passes \p rank as root rank to the underlying call.
/// This parameter is needed in functions like \c MPI_Gather.
///
/// @param rank The root rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto root(int rank) {
    return internal::RootDataBuffer(rank);
}

/// @brief Passes \p rank as root rank to the underlying call.
/// This parameter is needed in functions like \c MPI_Gather.
///
/// @param rank The root rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto root(size_t rank) {
    return root(asserting_cast<int>(rank));
}

/// @brief Passes \p rank as destination rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Send.
///
/// @param rank The destination rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto destination(int rank) {
    return internal::RankDataBuffer<internal::RankType::value, internal::ParameterType::destination>(rank);
}

/// @brief Passes \p rank as destination rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Send.
///
/// @param rank The destination rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto destination(size_t rank) {
    return destination(asserting_cast<int>(rank));
}

/// @brief Passes \c MPI_PROC_NULL as destination rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Send.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto destination(internal::rank_null_t) {
    return internal::RankDataBuffer<internal::RankType::null, internal::ParameterType::destination>{};
}

/// @brief Passes \p rank as source rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Recv.
///
/// @param rank The source rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto source(int rank) {
    return internal::RankDataBuffer<internal::RankType::value, internal::ParameterType::source>(rank);
}

/// @brief Passes \p rank as source rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Recv.
///
/// @param rank The source rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto source(size_t rank) {
    return source(asserting_cast<int>(rank));
}

/// @brief Indicates to use \c MPI_ANY_SOURCE as source rank in the underlying call, i.e. accepting any rank as source
/// rank. This parameter is needed in point-to-point exchange routines like \c MPI_Recv.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto source(internal::rank_any_t) {
    return internal::RankDataBuffer<internal::RankType::any, internal::ParameterType::source>{};
}

/// @brief Passes \c MPI_PROC_NULL as source rank to the underlying call.
/// This parameter is needed in point-to-point exchange routines like \c MPI_Recv.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto source(internal::rank_null_t) {
    return internal::RankDataBuffer<internal::RankType::null, internal::ParameterType::source>{};
}

/// @brief Indicates to use \c MPI_ANY_TAG as tag in the underlying call.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto tag(internal::any_tag_t) {
    return internal::TagParam<internal::TagType::any>{};
}

/// @brief Passes \p value as tag to the underlying call.
///
/// @param value The tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto tag(int value) {
    return internal::TagParam<internal::TagType::value>{value};
}

/// @brief Converts the passed enum \p value to its integer representation and passes this value to the underlying call.
///
/// @param value The tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename EnumType, typename = std::enable_if_t<std::is_enum_v<EnumType>>>
inline auto tag(EnumType value) {
    static_assert(
        std::is_convertible_v<std::underlying_type_t<EnumType>, int>,
        "The underlying enum type must be implicitly convertible to int."
    );
    return tag(static_cast<int>(value));
}

/// @brief Indicates to use \c MPI_ANY_TAG as send tag in the underlying call.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_tag(internal::any_tag_t) {
    return internal::TagParam<internal::TagType::any, internal::ParameterType::send_tag>{};
}

/// @brief Passes \p value as send tag to the underlying call.
///
/// @param value The tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_tag(int value) {
    return internal::TagParam<internal::TagType::value, internal::ParameterType::send_tag>{value};
}

/// @brief Converts the passed enum \p value to its integer representation and passes this value to the underlying call.
///
/// @param value The send tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename EnumType, typename = std::enable_if_t<std::is_enum_v<EnumType>>>
inline auto send_tag(EnumType value) {
    static_assert(
        std::is_convertible_v<std::underlying_type_t<EnumType>, int>,
        "The underlying enum type must be implicitly convertible to int."
    );
    return send_tag(static_cast<int>(value));
}

/// @brief Indicates to use \c MPI_ANY_TAG as recv tag in the underlying call.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_tag(internal::any_tag_t) {
    return internal::TagParam<internal::TagType::any, internal::ParameterType::recv_tag>{};
}

/// @brief Passes \p value as recv tag to the underlying call.
///
/// @param value The tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_tag(int value) {
    return internal::TagParam<internal::TagType::value, internal::ParameterType::recv_tag>{value};
}

/// @brief Converts the passed enum \p value to its integer representation and passes this value to the underlying call.
///
/// @param value The recv tag value.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename EnumType, typename = std::enable_if_t<std::is_enum_v<EnumType>>>
inline auto recv_tag(EnumType value) {
    static_assert(
        std::is_convertible_v<std::underlying_type_t<EnumType>, int>,
        "The underlying enum type must be implicitly convertible to int."
    );
    return recv_tag(static_cast<int>(value));
}

/// @brief Passes a request handle to the underlying MPI call.
/// @param request The request handle.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto request(Request& request) {
    return internal::make_data_buffer<
        internal::ParameterType,
        internal::ParameterType::request,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize>(request);
}

/// @brief Passes a request from a \ref RequestPool to the underlying MPI call.
/// @param request The request handle.
/// @tparam IndexType The type of the index used by the \ref RequestPool for requests.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
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
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto request() {
    return internal::make_data_buffer<
        internal::ParameterType,
        internal::ParameterType::request,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize>(alloc_new<Request>);
}

/// @brief Passes the send mode parameter for point to point communication to the underlying call.
/// Pass any of the tags from the \c kamping::send_modes namespace.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename SendModeTag>
inline auto send_mode(SendModeTag) {
    return internal::SendModeParameter<SendModeTag>{};
}

/// @brief Passes a reduction operation to ther underlying call. Accepts function objects, lambdas, function pointers or
/// native \c MPI_Op as argument.
///
/// @tparam Op the type of the operation
/// @tparam Commutative tag whether the operation is commutative
/// @param op the operation
/// @param commute the commutativity tag
///     May be any instance of \c commutative, \c or non_commutative. Passing \c undefined_commutative is only
///     supported for builtin and native operations. This is used to streamline the interface so that the use does not
///     have to provide commutativity info when the operation is builtin.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Op, typename Commutative = ops::internal::undefined_commutative_tag>
internal::OperationBuilder<Op, Commutative>
op(Op&& op, Commutative commute = ops::internal::undefined_commutative_tag{}) {
    return internal::OperationBuilder<Op, Commutative>(std::forward<Op>(op), commute);
}

/// @brief Passes a container containing the value(s) to return on the first rank to \ref
/// kamping::Communicator::exscan().
///
/// @tparam Container Container type.
/// @param container Value(s) to return on the first rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename Container>
inline auto values_on_rank_0(Container&& container) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::values_on_rank_0,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::forward<Container>(container));
}

/// @brief Passes the data to be returned on the first rank in \c MPI_Exscan which is provided as an initializer list to
/// \ref kamping::Communicator::exscan().
///
/// @param values Value(s) to return on the first rank.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
template <typename T>
inline auto values_on_rank_0(std::initializer_list<T> values) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::values_on_rank_0,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize>(std::move(values));
}

/// @brief Passes \p send_type as send type to the underlying call.
///
/// @param send_type MPI_Datatype to use in the wrapped \c MPI operation.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_type(MPI_Datatype send_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(send_type));
}

/// @brief Indicates to deduce the send type in the underlying call and return it as part of underlying call's result
/// object.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Passes \p send_type, into which the send type deduced by KaMPIng will be written, to the underlying call.
/// The type will be stored at the location referred to by the provided reference.
///
/// @param send_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_type_out(MPI_Datatype& send_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(send_type);
}

/// @brief Passes \p recv_type as recv type to the underlying call.
///
/// @param recv_type MPI_Datatype to use in the wrapped \c MPI function.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_type(MPI_Datatype recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(recv_type));
}

/// @brief Indicates to deduce the receive type in the underlying call and return it as part of underlying call's result
/// object.
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Passes \p recv_type, into which the recv type deduced by KaMPIng will be written, to the underlying call.
/// The type will be stored at the location referred to by the provided reference.
///
/// @param recv_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto recv_type_out(MPI_Datatype& recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(recv_type);
}

/// @brief Passes \p send_recv_type as send/recv type to the underlying call.
/// (This parameter is in \c MPI routines such as \c MPI_Bcast, ... .)
///
/// @param send_recv_type MPI_Datatype to use in the wrapped \c MPI operation.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_type(MPI_Datatype send_recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::constant,
        internal::BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(std::move(send_recv_type));
}

/// @brief Indicates to deduce the send/recv type in the underlying call and return it as part of underlying call's
/// result object.
/// (This parameter is used in \c MPI routines such as \c MPI_Bcast, ... .)
///
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_type_out() {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(alloc_new<MPI_Datatype>);
}

/// @brief Passes \p send_recv_type, into which the send/recv type deduced by KaMPIng will be written, to the underlying
/// call. The type will be stored at the location referred to by the provided reference. (This parameter is used in \c
/// MPI routines such as \c MPI_Bcast, ... .)
///
/// @param send_recv_type Reference to the location at which the deduced MPI_Datatype will be stored.
/// @return The corresponding parameter object.
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
inline auto send_recv_type_out(MPI_Datatype& send_recv_type) {
    return internal::make_data_buffer_builder<
        internal::ParameterType::send_recv_type,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        MPI_Datatype>(send_recv_type);
}
/// @}

} // namespace params
using namespace params;
} // namespace kamping
