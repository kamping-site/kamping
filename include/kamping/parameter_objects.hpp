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
/// - LibAllocatedUniquePtrBasedBuffer
/// - LibAllocatedUniquePtrBasedBuffer
/// - MovedContainerBasedBuffer
/// provide memory to store the result of \c MPI calls and (intermediate information needed to complete an \c MPI call
/// like send displacements or receive counts/displacements etc. if the user has not yet provided them). The storage can
/// be either provided by the user or can be allocated by the library.
///

#pragma once

#include <cstddef>
#include <memory>
#include <mpi.h>
#include <type_traits>

#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_type_definitions.hpp"

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


///
/// @brief Object referring to a contiguous sequence of size objects.
///
/// Since KaMPI.ng needs to be C++17 compatible and std::span is part of C++20, we need our own implementation of the
/// above-described functionality.
/// @tparam T type for which the span is defined.
template <typename T>
struct Span {
    using value_type = T; ///< Value type of the underlying pointer
    const T* ptr;         ///< Pointer to the data referred to by Span.
    size_t   size;        ///< Number of elements of type T referred to by Span.
};


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
//     ///@brief Get access to the underlying read-only storage.
//     ///@return Span referring to the underlying read-only storage.
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
    ContainerBasedConstBuffer(const Container& container) : _container(container) {}

    /// @brief Get access to the underlying read-only storage.
    /// @return Span referring to the underlying read-only storage.
    Span<value_type> get() const {
        return {std::data(_container), _container.size()};
    }

private:
    const Container& _container; ///< Container which holds the actual data.
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

    ///@brief Constructor for UserAllocatedContainerBasedBuffer.
    /// param container Container providing storage for data that may be written.
    UserAllocatedContainerBasedBuffer(Container& cont) : _container(cont) {}

    ///@brief Request memory sufficient to hold at least \c size elements of \c value_type.
    ///
    /// If the underlying container does not provide enough memory it will be resized.
    ///@param size Number of elements for which memory is requested.
    ///@return Pointer to enough memory for \c size elements of type \c value_type.
    value_type* get_ptr(size_t size) {
        if (_container.size() < size)
            _container.resize(size);
        return _container.data();
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
    ///@brief Constructor for LibAllocatedContainerBasedBuffer.
    ///
    LibAllocatedContainerBasedBuffer() {}

    ///@brief Request memory sufficient to hold at least \c size elements of \c value_type.
    ///
    /// If the underlying container does not provide enough memory it will be resized.
    ///@param size Number of elements for which memory is requested.
    ///@return Pointer to enough memory for \c size elements of type \c value_type.
    value_type* get_ptr(size_t size) {
        _container.resize(size);
        return std::data(_container);
    }

    ///@brief Extract the underlying container. This will leave LibAllocatedContainerBasedBuffer in an unspecified
    /// state.
    ///
    ///@return Moves the underlying container out of the LibAllocatedContainerBasedBuffer.
    Container extract() {
        return std::move(_container);
    }

private:
    Container _container; ///< Container which holds the actual data.
};

///@brief Encapsulates rank of the root PE. This is needed for \c MPI collectives like \c MPI_Gather.
class Root {
public:
    static constexpr ParameterType parameter_type =
        ParameterType::root; ///< The type of parameter this object encapsulates.
    ///@ Constructor for Root.
    ///@param rank Rank of the root PE.
    Root(int rank) : _rank{rank} {}
    ///@brief Returns the rank of the root.
    ///@returns Rank of the root.
    int rank() const {
        return _rank;
    }

private:
    int _rank; ///< Rank of the root PE.
};


/// @brief Parameter wrapping an operation passed to reduce-like MPI collectives.
/// @tparam Op type of the operation (may be a function object or a lambda)
/// @tparam Commutative tag specifying if the operation is commutative
template <typename Op, typename Commutative>
class OperationBuilder {
public:
    static constexpr ParameterType parameter_type =
        ParameterType::op; ///< The type of parameter this object encapsulates.
    OperationBuilder(Op&& op, Commutative&&) : _op(op) {}
    template <typename T>
    auto build_operation() {
        static_assert(std::is_invocable_r_v<T, Op, T, T>, "Type of custom operation does not match.");
        return ReduceOperation<T, Op, Commutative>(std::move(_op), Commutative{});
    }

private:
    Op _op;
};

} // namespace internal

/// @}

} // namespace kamping
