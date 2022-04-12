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
/// The classes defined in this file serve as in, out and in/out parameters to the
/// \c MPI calls wrapped by KaMPI.ng.
/// The non-modifiable buffers (PtrBasedConstBuffer, ContainerBasedConstBuffer)
/// encapsulate input data like data to send or send counts needed for a lot of \c MPI calls. If the user already
/// computed additional information like the send displacements or receive counts for a collective operations that would
/// otherwise have to be computed by the library, these values can also be provided to the library via non-modifiable
/// buffers.
/// The modifiable buffers:
/// - UserAllocatedContainerBasedBuffer
/// - UserAllocatedUniquePtrBasedBuffer
/// - LibAllocatedContainerBasedBuffer
/// - LibAllocatedUniquePtrBasedBuffer
/// - MovedContainerBasedBuffer
/// provide memory to store the result of \c MPI calls and (intermediate information needed to complete an \c MPI call
/// like send displacements or receive counts/displacements etc. if the user has not yet provided them). The storage can
/// be either provided by the user or can be allocated by the library.
///

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_type_definitions.hpp"
#include "kamping/span.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Type used for tag dispatching.
///
/// This types needs to be used to select internal::LibAllocContainerBasedBuffer as buffer type.
template <typename Container>
struct NewContainer {};
/// @brief Type used for tag dispatching.
///
/// This types needs to be used to select internal::LibAllocUniquePtrBasedBuffer as buffer type.
template <typename T>
struct NewPtr {};

namespace internal {

//@todo enable once the tests have been written
///// @brief Constant buffer based on a pointer.
/////
///// PtrBasedConstBuffer wraps read-only buffer storage of type T and represents an input of ParameterType
///// type.
///// @tparam T type contained in the buffer.
///// @tparam ParameterType parameter type represented by this buffer.
// template <typename T, ParameterType type>
// class PtrBasedConstBuffer {
// public:
//     static constexpr ParameterType parameter_type         = type;  ///< The type of parameter this buffer represents.
//     static constexpr bool          is_modifiable = false; ///< Indicates whether the underlying storage is
//     modifiable. using value_type                             = T;     ///< Value type of the buffer.
//
//     PtrBasedConstBuffer(const T* ptr, size_t size) : _span{ptr, size} {}
//
//     /// @brief Get access to the underlying read-only storage.
//     /// @return Span referring to the underlying read-only storage.
//     Span<T> get() const {
//         return _span;
//     }
//
// private:
//     Span<T> _span; ///< Actual storage to which PtrBasedConstBuffer refers.
// };

/// @brief Constant buffer based on a container type.
///
/// ContainerBasedConstBuffer wraps read-only buffer storage provided by an std-like container like std::vector. The
/// Container type must provide \c data(), \c size() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType type>
class ContainerBasedConstBuffer {
public:
    static constexpr ParameterType parameter_type = type;  ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable  = false; ///< Indicates whether the underlying storage is modifiable.
    using value_type                              = typename Container::value_type; ///< Value type of the buffer.

    /// @brief Constructor for ContainerBasedConstBuffer.
    /// @param container Container holding the actual data.
    ContainerBasedConstBuffer(Container const& container) : _container(container) {}

    /// @brief Move constructor for ContainerBasedConstBuffer.
    ContainerBasedConstBuffer(ContainerBasedConstBuffer&&) = default;
    // move assignment operator is implicitly deleted as this buffer has a reference member

    /// @brief Copy constructor is deleted as buffers should only be moved.
    ContainerBasedConstBuffer(ContainerBasedConstBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    ContainerBasedConstBuffer& operator=(ContainerBasedConstBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage.
    size_t size() const {
        return _container.size();
    }

    /// @brief Get access to the underlying read-only storage.
    /// @return Span referring to the underlying read-only storage.
    Span<const value_type> get() const {
        return {std::data(_container), _container.size()};
    }

private:
    const Container& _container; ///< Container which holds the actual data.
};

/// @brief Empty buffer that can be used as default argument for optional buffer parameters.
/// @tparam ParameterType Parameter type represented by this pseudo buffer.
template <typename Data, ParameterType type>
class EmptyBuffer {
public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable =
        false;               ///< This pseudo buffer is not modifiable since it represents no actual buffer.
    using value_type = Data; ///< Value type of the buffer.

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage (always 0).
    size_t size() const {
        return 0;
    }

    /// @brief Returns a span containing a nullptr.
    /// @return Span containing a nullptr.
    Span<value_type> get() const {
        return {nullptr, 0};
    }
};

/// @brief Constant buffer for a single type, i.e., not a container.
///
/// SingleElementConstBuffer wraps a read-only value and is used instead of \ref ContainerBasedConstBuffer if only a
/// single element is sent or received and no container is needed.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType Parameter type represented by this buffer.
template <typename DataType, ParameterType type>
class SingleElementConstBuffer {
public:
    static constexpr ParameterType parameter_type = type;  ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable  = false; ///< Indicates whether the underlying storage is modifiable.
    using value_type                              = DataType; ///< Value type of the buffer.

    /// @brief Constructor for SingleElementConstBuffer.
    /// @param element Element holding that is wrapped.
    SingleElementConstBuffer(DataType const& element) : _element(element) {}

    /// @brief Move constructor for SingleElementConstBuffer.
    SingleElementConstBuffer(SingleElementConstBuffer&&) = default;
    // move assignment operator is implicitly deleted as this buffer has a reference member

    /// @brief Copy constructor is deleted as buffers should only be moved.
    SingleElementConstBuffer(SingleElementConstBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    SingleElementConstBuffer& operator=(SingleElementConstBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage (always 1).
    size_t size() const {
        return 1;
    }

    /// @brief Get access to the underlaying read-only value.
    /// @return Span referring to the underlying read-only storage.
    Span<const value_type> get() const {
        return {&_element, 1};
    }

private:
    DataType const& _element; ///< Reference to the actual data.
};

/// @brief Buffer based on a single element type that has been allocated by the user.
///
/// SingleElementModifiableBuffer wraps modifiable single-element buffer storage that has already been allocated by the
/// user.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename DataType, ParameterType type>
class SingleElementModifiableBuffer {
public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable  = true; ///< Indicates whether the underlying storage is modifiable.
    using value_type                              = DataType; ///< Value type of the buffer.

    /// @brief Constructor for SingleElementConstBuffer.
    /// @param element Element holding that is wrapped.
    SingleElementModifiableBuffer(DataType& element) : _element(element) {
        static_assert(
            !std::is_const_v<DataType>,
            "The underlying data type of a SingleElementModifiableBuffer must not be const.");
    }

    /// @brief Move constructor for SingleElementModifiableBuffer (implicitly deletes copy constructor/assignment
    /// operator).
    SingleElementModifiableBuffer(SingleElementModifiableBuffer&&) = default;
    // move assignment operator is implicitly deleted as this buffer has a reference member

    /// @brief Copy constructor is deleted as buffers should only be moved.
    SingleElementModifiableBuffer(SingleElementModifiableBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    SingleElementModifiableBuffer& operator=(SingleElementModifiableBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Does nothing but assert that only size 1 is requested.
    ///
    /// @param size The size that this "container" is expected to have after the call.
    void resize(size_t size) const {
        KASSERT(size == 1ul, "Single element buffers must hold exactly one element.");
    }

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage (always 1).
    size_t size() const {
        return 1;
    }

    /// @brief Get writable access to the underlaying value.
    /// @return Reference to the underlying storage.
    Span<value_type> get() const {
        return {&_element, 1};
    }

private:
    DataType& _element; ///< (Writable) reference to the actual data.
};

/// @brief Struct containing some definitions used by all modifiable buffers.
///
/// @tparam ParameterType (parameter) type represented by this buffer
/// @tparam is_consumable_ indicates whether this buffer already contains useable data
template <ParameterType type>
struct BufferParameterType {
    static constexpr ParameterType parameter_type = type; ///< ParameterType which the buffer represents.
    static constexpr bool          is_modifiable  = true; ///< Indicates whether the underlying storage is modifiable.
};

/// @brief Buffer based on a container type that has been allocated by the user (but may be resized if the provided
/// space is not sufficient).
///
/// UserAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like std::vector
/// that has already been allocated by the user. The Container type must provide \c data(), \c size() and \c resize()
/// and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType parameter_type>
class UserAllocatedContainerBasedBuffer : public BufferParameterType<parameter_type> {
public:
    using value_type = typename Container::value_type; ///< Value type of the buffer.

    /// @brief Constructor for UserAllocatedContainerBasedBuffer.
    /// param container Container providing storage for data that may be written.
    UserAllocatedContainerBasedBuffer(Container& cont) : _container(cont) {}

    /// @brief Move constructor for UserAllocatedContainerBasedBuffer (implicitly deletes copy constructor/assignment
    /// operator).
    UserAllocatedContainerBasedBuffer(UserAllocatedContainerBasedBuffer&&) = default;
    // move assignment operator is implicitly deleted as this buffer has a reference member

    /// @brief Copy constructor is deleted as buffers should only be moved.
    UserAllocatedContainerBasedBuffer(UserAllocatedContainerBasedBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    UserAllocatedContainerBasedBuffer& operator=(UserAllocatedContainerBasedBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Resizes container such that it holds exactly \c size elements of \c value_type if the \c Container is not
    /// a \c Span.
    ///
    /// This function calls \c resize on the container if the container is of type \c Span. If the container is a \c
    /// Span,  KaMPI.ng assumes that the memory is managed by the user and that resizing is not wanted. In this case it
    /// is \c KASSERTed that the memory provided by the span is sufficient. Whether new memory is allocated and/or data
    /// is  copied depends in the implementation of the container.
    ///
    /// @param size Size the container is resized to if it is not a \c Span.
    void resize(size_t size) {
        if constexpr (!std::is_same_v<Container, Span<value_type>>) {
            _container.resize(size);
        } else {
            KASSERT(_container.size() >= size, "Span cannot be resized and is smaller than the requested size.");
        }
    }

    /// @brief Get writable access to the underlaying container.
    /// @return Pointer to the underlying container.
    value_type* data() {
        return _container.data();
    }

    /// @brief Get writable access to the underlaying container.
    /// @return Reference to the underlying container.
    Span<value_type> get() {
        return {_container.data(), _container.size()};
    }

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage.
    size_t size() const {
        return _container.size();
    }

private:
    Container& _container; ///< Container which holds the actual data.
};

/// @brief Buffer based on a container type that will be allocated by the library (using the container's allocator)
///
/// LibAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like std::vector
/// that will be allocated by KaMPI.ng. The Container type must provide \c data(), \c size() and \c resize() and
/// expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType type>
class LibAllocatedContainerBasedBuffer : public BufferParameterType<type> {
public:
    using value_type = typename Container::value_type; ///< Value type of the buffer.

    /// @brief Constructor for LibAllocatedContainerBasedBuffer.
    LibAllocatedContainerBasedBuffer() = default;

    /// @brief Move constructor for LibAllocatedContainerBasedBuffer.
    LibAllocatedContainerBasedBuffer(LibAllocatedContainerBasedBuffer&&) = default;

    /// @brief Move assignment operator for LibAllocatedContainerBasedBuffer.
    LibAllocatedContainerBasedBuffer& operator=(LibAllocatedContainerBasedBuffer&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    LibAllocatedContainerBasedBuffer(LibAllocatedContainerBasedBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    LibAllocatedContainerBasedBuffer& operator=(LibAllocatedContainerBasedBuffer const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Resizes container such that it holds exactly \c size elements of \c value_type if the \c Container is not
    /// a \c Span.
    ///
    /// This function calls \c resize on the container if the container is of type \c Span. If the container is a \c
    /// Span,  KaMPI.ng assumes that the memory is managed by the user and that resizing is not wanted. In this case it
    /// is \c KASSERTed that the memory provided by the span is sufficient. Whether new memory is allocated and/or data
    /// is  copied depends in the implementation of the container.
    ///
    /// @param size Size the container is resized to if it is not a \c Span.
    void resize(size_t size) {
        if constexpr (!std::is_same_v<Container, Span<value_type>>) {
            _container.resize(size);
        } else {
            KASSERT(_container.size() >= size, "Span cannot be resized and is smaller than the requested size.");
        }
    }

    /// @brief Get writable access to the underlaying container.
    /// @return Reference to the underlying container.
    Span<value_type> get() {
        return {_container.data(), _container.size()};
    }

    /// @brief Get writable access to the underlaying container.
    /// @return Reference to the underlying container.
    value_type* data() {
        return _container.data();
    }

    /// @brief Extract the underlying container. This will leave LibAllocatedContainerBasedBuffer in an unspecified
    /// state.
    ///
    /// @return Moves the underlying container out of the LibAllocatedContainerBasedBuffer.
    Container extract() {
        return std::move(_container);
    }

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage.
    size_t size() const {
        return _container.size();
    }

private:
    Container _container; ///< Container which holds the actual data.
};

/// @brief Encapsulates the recv count in a collective operation.
/// @tparam Value type or reference type, depending on whether this is an input- our output parameter.
template <typename T>
class RecvCount {
public:
    static_assert(
        std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, int>,
        "Underlaying recv count value type must be int.");

    static constexpr ParameterType parameter_type =
        ParameterType::recv_count; ///< The tag of the parameter that this object encapsulates.
    static constexpr bool is_modifiable =
        !std::is_const_v<T> && std::is_reference_v<T>; ///< Whether this is an input parameter or an output parameter.

    /// @brief Constructor for encapsulated recv count.
    /// @param recv_count Encapsulated recv count.
    RecvCount(T recv_count) : _recv_count{recv_count} {}

    /// @brief Move constructor for RecvCount.
    RecvCount(RecvCount&&) = default;

    /// @brief Move assignment operator for RecvCount.
    RecvCount& operator=(RecvCount&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    RecvCount(RecvCount const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    RecvCount& operator=(RecvCount const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Returns the encapsulated recv count.
    /// @returns The encapsulated recv count.
    int recv_count() const {
        return _recv_count; // type of _recv_count is always based on int
    }

    /// @brief Updates the recv count (only if used to wrap an output parameter).
    /// @param recv_count New recv count.
    template <bool modifiable = is_modifiable, std::enable_if_t<modifiable, bool> = true>
    void set_recv_count(int const recv_count) {
        _recv_count = recv_count;
    }

    /// @brief Returns the encapsulated recv count. To be used when the receive count is part of MPIResult.
    /// @return The encapsulate recv count.
    int extract() const {
        return _recv_count; // type of _recv_count is always based on int
    }

private:
    T _recv_count; ///< Encapsulated recv count.
};

/// @brief Encapsulates rank of the root PE. This is needed for \c MPI collectives like \c MPI_Gather.
class Root {
public:
    static constexpr ParameterType parameter_type =
        ParameterType::root; ///< The type of parameter this object encapsulates.

    /// @ Constructor for Root.
    /// @param rank Rank of the root PE.
    Root(size_t rank) : _rank{rank} {}

    /// @ Constructor for Root.
    /// @param rank Rank of the root PE.
    Root(int rank) : _rank{asserting_cast<size_t>(rank)} {}

    /// @brief Move constructor for Root.
    Root(Root&&) = default;

    /// @brief Move assignment operator for Root.
    Root& operator=(Root&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    Root(Root const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    Root& operator=(Root const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Returns the rank of the root as `size_t`.
    /// @returns Rank of the root as `size_t`.
    size_t rank() const {
        return _rank;
    }

    /// @brief Returns the rank of the root as `int`.
    /// @returns Rank of the root as `int`.
    int rank_signed() const {
        return asserting_cast<int>(_rank);
    }

private:
    size_t _rank; ///< Rank of the root PE.
};

/// @brief Parameter wrapping an operation passed to reduce-like MPI collectives.
/// This wraps an MPI operation without the argument of the operation specified. This enables the user to construct such
/// wrapper using the parameter factory \c kamping::op without passing the type of the operation.
/// The library developer may then construct the actual operation wrapper with a given type later.
///
/// @tparam Op type of the operation (may be a function object or a lambda)
/// @tparam Commutative tag specifying if the operation is commutative
template <typename Op, typename Commutative>
class OperationBuilder {
public:
    static constexpr ParameterType parameter_type =
        ParameterType::op; ///< The type of parameter this object encapsulates.

    /// @brief constructs an Operation builder
    /// @param op the operation
    /// @param commutative_tag tag indicating if the operation is commutative (see \c kamping::op for details)
    OperationBuilder(Op&& op, Commutative commutative_tag [[maybe_unused]]) : _op(op) {}

    /// @brief Move constructor for OperationsBuilder.
    OperationBuilder(OperationBuilder&&) = default;

    /// @brief Move assignment operator for OperationsBuilder.
    OperationBuilder& operator=(OperationBuilder&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    OperationBuilder(OperationBuilder const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    OperationBuilder& operator=(OperationBuilder const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief constructs an operation for the given type T
    /// @tparam T argument type of the reduction operation
    template <typename T>
    [[nodiscard]] auto build_operation() {
        static_assert(std::is_invocable_r_v<T, Op, T&, T&>, "Type of custom operation does not match.");
        return ReduceOperation<T, Op, Commutative>(std::forward<Op>(_op), Commutative{});
    }

private:
    Op _op; ///< the operation which is encapsulated
};

} // namespace internal

/// @}

} // namespace kamping
