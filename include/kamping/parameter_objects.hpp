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
/// \c MPI calls wrapped by KaMPIng.
///
/// The non-modifiable buffers encapsulate input data like data to send or send counts needed for a lot of \c MPI calls.
/// If the user already computed additional information like the send displacements or receive counts for a collective
/// operations that would otherwise have to be computed by the library, these values can also be provided to the library
/// via non-modifiable buffers.
///
/// The modifiable buffers provide memory to store the result of \c MPI calls and
/// (intermediate information needed to complete an \c MPI call like send displacements or receive counts/displacements
/// etc. if the user has not yet provided them). The storage can be either provided by the user or can be allocated by
/// the library.
///

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_type_definitions.hpp"
#include "kamping/span.hpp"
#include "kassert/kassert.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Type used for tag dispatching.
///
/// This types needs to be used to select internal::LibAllocContainerBasedBuffer as buffer type.
template <typename Container>
struct NewContainer {};

namespace internal {
/// @brief Helper to decide if data type has \c .data() method.
/// @return \c std::true_type if class has \c .data() method and \c std::false_type otherwise.
template <typename, typename = void>
struct has_data_member : std::false_type {};

/// @brief Helper to decide if data type has \c .data() method.
/// @return \c std::true_type if class has \c .data() method and \c std::false_type otherwise.
template <typename T>
struct has_data_member<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {};

/// @brief Boolean value helping to decide if data type has \c .data() method.
/// @return \c true if class has \c .data() method and \c false otherwise.
template <typename T>
inline constexpr bool has_data_member_v = has_data_member<T>::value;

/// @brief Enum to specify whether a buffer is modifiable
enum class BufferModifiability { modifiable, constant };
/// @brief Enum to specify whether a buffer owns its data
enum class BufferOwnership { owning, referencing };
/// @brief Enum to specify whether a buffer is allocated by the library or the user
enum class BufferAllocation { lib_allocated, user_allocated };

/// @brief Wrapper to get the value type of a non-container type (aka the type itself).
/// @tparam has_value_type_member Whether `T` has a value_type member
/// @tparam T The type to get the value_type of
template <bool has_value_type_member /*= false */, typename T>
class ValueTypeWrapper {
public:
    using value_type = T; ///< The value type of T.
};

/// @brief Wrapper to get the value type of a container type.
/// @tparam T The type to get the value_type of
template <typename T>
class ValueTypeWrapper</*has_value_type_member =*/true, T> {
public:
    using value_type = typename T::value_type; ///< The value type of T.
};

/// @brief Data buffer used for named parameters.
///
/// DataBuffer wraps all buffer storages provided by an std-like container like std::vector or single values. A
/// Container type must provide \c data(), \c size() and expose the type definition \c value_type.
/// @tparam MemberType Container or data type on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
/// @tparam modifiability `modifiable` if a KaMPIng operation is allowed to
/// modify the underlying container. `constant` otherwise.
/// @tparam ownership `owning` if the buffer should hold the actual container.
/// `referencing` if only a reference to an existing container should be held.
/// @tparam allocation `lib_allocated` if the buffer was allocated by the library,
/// `user_allocated` if it was allocated by the user.
template <
    typename MemberType, ParameterType type, BufferModifiability modifiability, BufferOwnership ownership,
    BufferAllocation allocation = BufferAllocation::user_allocated>
class DataBuffer {
public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable =
        modifiability == BufferModifiability::modifiable; ///< Indicates whether the underlying storage is modifiable.
    static constexpr bool is_single_element =
        !has_data_member_v<MemberType>; ///<`true` if the DataBuffer represents a singe element, `false` if the
                                        ///< DataBuffer represents a container.
    using MemberTypeWithConst =
        std::conditional_t<is_modifiable, MemberType, MemberType const>; ///< The ContainerType as const or
                                                                         ///< non-const depending on
                                                                         ///< modifiability.
    using MemberTypeWithConstAndRef = std::conditional_t<
        ownership == BufferOwnership::owning, MemberTypeWithConst,
        MemberTypeWithConst&>; ///< The ContainerType as const or non-const (see ContainerTypeWithConst) and
                               ///< reference or non-reference depending on ownership.

    using value_type =
        typename ValueTypeWrapper<!is_single_element, MemberType>::value_type; ///< Value type of the buffer.
    using value_type_with_const =
        std::conditional_t<is_modifiable, value_type, value_type const>; ///< Value type as const or non-const depending
                                                                         ///< on modifiability

    /// @brief Constructor for referencing ContainerBasedBuffer.
    /// @param container Container holding the actual data.
    template <bool enabled = ownership == BufferOwnership::referencing, std::enable_if_t<enabled, bool> = true>
    DataBuffer(MemberTypeWithConst& container) : _data(container) {}

    /// @brief Constructor for owning ContainerBasedBuffer.
    /// @param container Container holding the actual data.
    template <bool enabled = ownership == BufferOwnership::owning, std::enable_if_t<enabled, bool> = true>
    DataBuffer(MemberType container) : _data(std::move(container)) {}

    /// @brief Constructor for lib allocated ContainerBasedBuffer.
    template <bool enabled = allocation == BufferAllocation::lib_allocated, std::enable_if_t<enabled, bool> = true>
    DataBuffer() : _data() {
        static_assert(ownership == BufferOwnership::owning, "Lib allocated buffers must be owning");
        static_assert(is_modifiable, "Lib allocated buffers must be modifiable");
    }

    /// @brief Move constructor.
    DataBuffer(DataBuffer&&) = default;

    /// @brief Move assignment operator.
    DataBuffer& operator=(DataBuffer&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    DataBuffer(DataBuffer const&) = delete;

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    DataBuffer& operator=(DataBuffer const&) = delete;

    /// @brief Get the number of elements in the underlying storage.
    /// @return Number of elements in the underlying storage.
    size_t size() const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get the size of a buffer that has already been extracted.", assert::normal);
#endif
        if constexpr (is_single_element) {
            return 1;
        } else {
            return _data.size();
        }
    }

    /// @brief Resizes the underlying container such that it holds exactly \c size elements of \c value_type if the \c
    /// MemberType is not a \c Span or a single elements.
    ///
    /// This function calls \c resize on the container if the container is not of type \c Span or a single value. If the
    /// container is a \c Span,  KaMPIng assumes that the memory is managed by the user and that resizing is not wanted.
    /// In this case it is \c KASSERTed that the memory provided by the span is sufficient. If the buffer stores only a
    /// single value, it is KASSERTed that the requested size is exactly 1. Whether new memory is
    /// allocated and/or data is copied depends in the implementation of the container.
    ///
    /// @param size Size the container is resized to if it is not a \c Span.
    void resize(size_t size) {
        // This works because in template classes, only functions that are actually called are instantiated
        // Technically not needed here because _data is const in this case, so we can't call resize() anyways. But this
        // gives a nicer error message.
        static_assert(is_modifiable, "Trying to resize a constant DataBuffer");
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot resize a buffer that has already been extracted.", assert::normal);
#endif
        if constexpr (is_single_element) {
            KASSERT(size == 1u, "Single element buffers must hold exactly one element.");
        } else if constexpr (std::is_same_v<MemberType, Span<value_type>>) {
            KASSERT(this->size() >= size, "Span cannot be resized and is smaller than the requested size.");
        } else {
            _data.resize(size);
        }
    }

    /// @brief Get const access to the underlying container.
    /// @return Pointer to the underlying container.
    value_type const* data() const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get a pointer to a buffer that has already been extracted.", assert::normal);
#endif
        if constexpr (is_single_element) {
            return &_data;
        } else {
            return std::data(_data);
        }
    }

    /// @brief Get access to the underlying container.
    /// @return Pointer to the underlying container.
    value_type_with_const* data() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get a pointer to a buffer that has already been extracted.", assert::normal);
#endif
        if constexpr (is_single_element) {
            return &_data;
        } else {
            return std::data(_data);
        }
    }

    /// @brief Get read-only access to the underlying storage.
    /// @return Span referring the underlying storage.
    Span<value_type const> get() const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get a buffer that has already been extracted.", assert::normal);
#endif
        return {this->data(), this->size()};
    }

    /// @brief Get access to the underlying storage.
    /// @return Span referring to the underlying storage.
    Span<value_type_with_const> get() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get a buffer that has already been extracted.", assert::normal);
#endif
        return {this->data(), this->size()};
    }

    /// @brief Get the single element wrapped by this object.
    /// @return The single element wrapped by this object.
    template <bool enabled = is_single_element, std::enable_if_t<enabled, bool> = true>
    value_type const get_single_element() const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get an element from a buffer that has already been extracted.", assert::normal);
#endif
        return _data;
    }

    /// @brief Provides access to the underlying data.
    /// @return A reference to the data.
    MemberType const& underlying() const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot get a buffer that has already been extracted.", assert::normal);
#endif
        return _data;
    }

    /// @brief Extract the underlying container. This will leave the DataBuffer in an unspecified
    /// state.
    ///
    /// @return Moves the underlying container out of the DataBuffer.
    MemberTypeWithConst extract() {
        // This works because in template classes, only functions that are actually called are instantiated
        static_assert(
            allocation == BufferAllocation::lib_allocated,
            "extract() must only be called on library allocated DataBuffers");
        static_assert(
            ownership == BufferOwnership::owning, "Moving out of a reference should not be done because it would leave "
                                                  "a users container in an unspecified state.");
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, "Cannot extract a buffer that has already been extracted.", assert::normal);
        is_extracted = true;
#endif
        return std::move(_data);
    }

private:
    MemberTypeWithConstAndRef _data; ///< Container which holds the actual data.
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    bool is_extracted = false; ///< Has the container been extracted and is therefore in an invalid state?
#endif
};

/// @brief Constant buffer based on a container type.
///
/// ContainerBasedConstBuffer wraps read-only buffer storage provided by an std-like container like std::vector. The
/// Container type must provide \c data(), \c size() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType type>
using ContainerBasedConstBuffer =
    DataBuffer<Container, type, BufferModifiability::constant, BufferOwnership::referencing>;

/// @brief Read-only buffer owning a container type passed to it.
///
/// ContainerBasedOwningBuffer wraps read-only buffer storage provided by an std-like container like std::vector.
/// This is the owning variant of \ref ContainerBasedConstBuffer. The Container type must provide \c data(), \c
/// size() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType type>
using ContainerBasedOwningBuffer = DataBuffer<Container, type, BufferModifiability::constant, BufferOwnership::owning>;

/// @brief Buffer based on a container type that has been allocated by the user (but may be resized if the provided
/// space is not sufficient).
///
/// UserAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like
/// std::vector that has already been allocated by the user. The Container type must provide \c data(), \c size()
/// and \c resize() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType parameter_type>
using UserAllocatedContainerBasedBuffer =
    DataBuffer<Container, parameter_type, BufferModifiability::modifiable, BufferOwnership::referencing>;

/// @brief Buffer based on a container type that will be allocated by the library (using the container's allocator)
///
/// LibAllocatedContainerBasedBuffer wraps modifiable buffer storage provided by an std-like container like
/// std::vector that will be allocated by KaMPIng. The Container type must provide \c data(), \c size() and \c
/// resize() and expose the type definition \c value_type. type.
/// @tparam Container Container on which this buffer is based.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename Container, ParameterType type>
using LibAllocatedContainerBasedBuffer = DataBuffer<
    Container, type, BufferModifiability::modifiable, BufferOwnership::owning, BufferAllocation::lib_allocated>;

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

    /// @brief Get a nullptr.
    /// @return nullptr.
    value_type const* data() const {
        return nullptr;
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
using SingleElementConstBuffer =
    DataBuffer<DataType, type, BufferModifiability::constant, BufferOwnership::referencing>;

/// @brief Buffer for a single element, which is not a container. The element is owned by the buffer.
///
/// SingleElementOwningBuffer wraps a read-only value and takes ownership of it. It is the owning variant of \ref
/// SingleElementConstBuffer.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType Parameter type represented by this buffer.
template <typename DataType, ParameterType type>
using SingleElementOwningBuffer = DataBuffer<DataType, type, BufferModifiability::constant, BufferOwnership::owning>;

/// @brief Buffer based on a single element type that has been allocated by the library.
///
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename DataType, ParameterType type>
using LibAllocatedSingleElementBuffer = DataBuffer<
    DataType, type, BufferModifiability::modifiable, BufferOwnership::owning, BufferAllocation::lib_allocated>;

/// @brief Buffer based on a single element type that has been allocated by the user.
///
/// SingleElementModifiableBuffer wraps modifiable single-element buffer storage that has already been allocated by
/// the user.
/// @tparam DataType Type of the element wrapped.
/// @tparam ParameterType parameter type represented by this buffer.
template <typename DataType, ParameterType type>
using SingleElementModifiableBuffer =
    DataBuffer<DataType, type, BufferModifiability::modifiable, BufferOwnership::referencing>;

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
/// This wraps an MPI operation without the argument of the operation specified. This enables the user to construct
/// such wrapper using the parameter factory \c kamping::op without passing the type of the operation. The library
/// developer may then construct the actual operation wrapper with a given type later.
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
