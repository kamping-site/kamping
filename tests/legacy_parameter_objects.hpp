// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file This file contains the old separate container buffers. These have been replaced by one generic class
/// DataBuffer but are still used in a lot of the old tests. They are exact copies of what was used for the parameter
/// factories before.

#pragma once

#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping::internal {

/// @brief Constant buffer based on a container type.
///
/// ContainerBasedConstBuffer wraps read-only buffer storage provided by an std-like container like std::vector. The
/// Container type must provide \c data(), \c size() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename Container, ParameterType parameter_type, BufferType buffer_type>
using ContainerBasedConstBuffer =
    DataBuffer<Container, parameter_type, BufferModifiability::constant, BufferOwnership::referencing, buffer_type>;

/// @brief Read-only buffer owning a container type passed to it.
///
/// ContainerBasedOwningBuffer wraps read-only buffer storage provided by an std-like container like std::vector.
/// This is the owning variant of \ref ContainerBasedConstBuffer. The Container type must provide \c data(), \c
/// size() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename Container, ParameterType parameter_type, BufferType buffer_type>
using ContainerBasedOwningBuffer =
    DataBuffer<Container, parameter_type, BufferModifiability::constant, BufferOwnership::owning, buffer_type>;

/// @brief Buffer based on a container type that has been allocated by the user (but may be resized if the provided
/// space is not sufficient).
///
/// UserAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like
/// std::vector that has already been allocated by the user. The Container type must provide \c data(), \c size()
/// and \c resize() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename Container, ParameterType parameter_type, BufferType buffer_type>
using UserAllocatedContainerBasedBuffer =
    DataBuffer<Container, parameter_type, BufferModifiability::modifiable, BufferOwnership::referencing, buffer_type>;

/// @brief Buffer based on a container type that will be allocated by the library (using the container's allocator)
///
/// LibAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like
/// std::vector that will be allocated by KaMPIng. The Container type must provide \c data(), \c size() and \c
/// resize() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename Container, ParameterType parameter_type, BufferType buffer_type>
using LibAllocatedContainerBasedBuffer = DataBuffer<
    Container, parameter_type, BufferModifiability::modifiable, BufferOwnership::owning, buffer_type,
    BufferAllocation::lib_allocated>;

/// @brief Constant buffer for a single type, i.e., not a container.
///
/// SingleElementConstBuffer wraps a read-only value and is used instead of \ref ContainerBasedConstBuffer if only a
/// single element is sent or received and no container is needed.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType Parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename DataType, ParameterType parameter_type, BufferType buffer_type>
using SingleElementConstBuffer =
    DataBuffer<DataType, parameter_type, BufferModifiability::constant, BufferOwnership::referencing, buffer_type>;

/// @brief Buffer for a single element, which is not a container. The element is owned by the buffer.
///
/// SingleElementOwningBuffer wraps a read-only value and takes ownership of it. It is the owning variant of \ref
/// SingleElementConstBuffer.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType Parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename DataType, ParameterType parameter_type, BufferType buffer_type>
using SingleElementOwningBuffer =
    DataBuffer<DataType, parameter_type, BufferModifiability::constant, BufferOwnership::owning, buffer_type>;

/// @brief Buffer based on a single element type that has been allocated by the library.
///
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename DataType, ParameterType parameter_type, BufferType buffer_type>
using LibAllocatedSingleElementBuffer = DataBuffer<
    DataType, parameter_type, BufferModifiability::modifiable, BufferOwnership::owning, buffer_type,
    BufferAllocation::lib_allocated>;

/// @brief Buffer based on a single element type that has been allocated by the user.
///
/// SingleElementModifiableBuffer wraps modifiable single-element buffer storage that has already been allocated by
/// the user.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam buffer_type Type of the buffer, i.e., in, out, or in_out.
template <typename DataType, ParameterType parameter_type, BufferType buffer_type>
using SingleElementModifiableBuffer =
    DataBuffer<DataType, parameter_type, BufferModifiability::modifiable, BufferOwnership::referencing, buffer_type>;

} // namespace kamping::internal
