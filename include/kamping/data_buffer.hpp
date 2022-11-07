// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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
#include "kamping/named_parameter_types.hpp"
#include "kamping/span.hpp"
#include "kassert/kassert.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {

/// @brief Boolean value helping to decide if type has a \c value_type member type.
/// @return \c true if class has \c value_type method and \c false otherwise.
template <typename, typename = void>
static constexpr bool has_value_type_v = false;

/// @brief Boolean value helping to decide if type has a \c value_type member type.
/// @return \c true if class has \c value_type method and \c false otherwise.
template <typename T>
static constexpr bool has_value_type_v<T, std::void_t<typename T::value_type>> = true;

/// @brief Type trait to check if a type is an instance of a templated type.
///
/// based on https://stackoverflow.com/a/31763111
/// @tparam T The concrete type.
/// @tparam Template The type template.
/// @return \c true if the type is an instance and \c false otherwise.
template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

/// @brief Type trait to check if a type is an instance of a templated type.
///
/// based on https://stackoverflow.com/a/31763111
///
/// A little note on how this works:
/// - consider <tt>is_specialization<std::vector<bool, my_alloc>, std::vector></tt>
/// - this gets template matched with the following specialization such that
///    - <tt>Template = template<T...> std::vector<T...></tt>
///    - <tt>Args... = bool, my_alloc</tt>
/// - but this may only be matched in the case that <tt>Template<Args...> = std::vector<bool, my_alloc></tt>
/// @tparam T The concrete type.
/// @tparam Template the type template
/// @return \c true if the type is an instance and \c false otherwise.
template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

/// @brief Boolean value helping to check if a type is an instance of \c std::vector<bool>.
/// @tparam T The type.
/// @return \c true if \c T is an template instance of \c std::vector<bool>, \c false otherwise.
template <typename T, typename = void>
static constexpr bool is_vector_bool_v = false;

/// @brief Boolean value helping to check if a type is an instance of \c std::vector<bool>.
/// This catches the edge case of elements which do not have a value type, they can not be a vector bool.
///
/// @tparam T The type.
/// @return \c true if \T is an template instance of \c std::vector<bool>, \c false otherwise.
template <typename T>
static constexpr bool is_vector_bool_v<
    T,
    typename std::enable_if<!has_value_type_v<std::remove_cv_t<std::remove_reference_t<T>>>>::type> = false;

/// @brief Boolean value helping to check if a type is an instance of \c std::vector<bool>.
/// @tparam T The type.
/// @return \c true if \T is an template instance of \c std::vector<bool>, \c false otherwise.
template <typename T>
static constexpr bool
    is_vector_bool_v<T, typename std::enable_if<has_value_type_v<std::remove_cv_t<std::remove_reference_t<T>>>>::type> =
        is_specialization<std::remove_cv_t<std::remove_reference_t<T>>, std::vector>::value&&
            std::is_same_v<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type, bool>;

} // namespace internal

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
/// @brief Enum to specify whether a buffer is an in buffer of an out
/// buffer. Out buffer will be used to directly write the result to.
enum class BufferType { in_buffer, out_buffer, in_out_buffer };

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

/// @brief The set of parameter types that must be of type `int`
constexpr std::array int_parameter_types{
    ParameterType::recv_counts, ParameterType::send_counts, ParameterType::recv_displs, ParameterType::send_displs};

/// @brief Checks whether buffers of a given type should have `value_type` `int`.
///
/// @param parameter_type The parameter type to check.
///
/// @return `true` if parameter_type should be of type `int`, `false` otherwise.
inline constexpr bool is_int_type(ParameterType parameter_type) {
    for (ParameterType int_parameter_type: int_parameter_types) {
        if (parameter_type == int_parameter_type) {
            return true;
        }
    }
    return false;
}

/// @brief Data buffer used for named parameters.
///
/// DataBuffer wraps all buffer storages provided by an std-like container like std::vector or single values. A
/// Container type must provide \c data(), \c size() and expose the type definition \c value_type.
/// @tparam MemberType Container or data type on which this buffer is based.
/// @tparam parameter_type_param Parameter type represented by this buffer.
/// @tparam modifiability `modifiable` if a KaMPIng operation is allowed to
/// modify the underlying container. `constant` otherwise.
/// @tparam ownership `owning` if the buffer should hold the actual container.
/// `referencing` if only a reference to an existing container should be held.
/// @tparam buffer_type_param Type of buffer, i.e., \c in_buffer, \c out_buffer, or \c in_out_buffer.
/// @tparam allocation `lib_allocated` if the buffer was allocated by the library,
/// `user_allocated` if it was allocated by the user.
template <
    typename MemberType,
    ParameterType       parameter_type_param,
    BufferModifiability modifiability,
    BufferOwnership     ownership,
    BufferType          buffer_type_param,
    BufferAllocation    allocation = BufferAllocation::user_allocated>
class DataBuffer {
public:
    static constexpr ParameterType parameter_type =
        parameter_type_param; ///< The type of parameter this buffer represents.

    static constexpr BufferType buffer_type = buffer_type_param; ///< The type of the buffer, i.e., in, out, or in_out.

    /// @brief \c true if the buffer is an out or in/out buffer that results will be written to and \c false
    /// otherwise.
    static constexpr bool is_out_buffer =
        (buffer_type_param == BufferType::out_buffer || buffer_type_param == BufferType::in_out_buffer);

    /// @brief Indicates whether the buffer is allocated by KaMPIng.
    static constexpr bool is_lib_allocated = allocation == BufferAllocation::lib_allocated;

    static constexpr bool is_modifiable =
        modifiability == BufferModifiability::modifiable; ///< Indicates whether the underlying storage is modifiable.
    static constexpr bool is_single_element =
        !has_data_member_v<MemberType>; ///<`true` if the DataBuffer represents a singe element, `false` if the
                                        ///< DataBuffer represents a container.
    using MemberTypeWithConst =
        std::conditional_t<is_modifiable, MemberType, MemberType const>; ///< The ContainerType as const or
                                                                         ///< non-const depending on
                                                                         ///< modifiability.

    // We can not do the check for std::vector<bool> here, because to use a DataBuffer of std::vector<bool> as an unused
    // default parameter is allowed, as long the buffer is never used. Therefore the check for std::vector<bool> happens
    // only when the underlying member is actually accessed and the corresponding accessor method is instantiated.
    // static_assert(
    //     !is_vector_bool_v<MemberType>,
    //     "Buffers based on std::vector<bool> are not supported, use std::vector<kamping::kabool> instead.");

    using MemberTypeWithConstAndRef = std::conditional_t<
        ownership == BufferOwnership::owning,
        MemberTypeWithConst,
        MemberTypeWithConst&>; ///< The ContainerType as const or non-const (see ContainerTypeWithConst) and
                               ///< reference or non-reference depending on ownership.

    using value_type =
        typename ValueTypeWrapper<!is_single_element, MemberType>::value_type; ///< Value type of the buffer.
    // Logical implication: is_int_type(type) => std::is_same_v<value_type, int>
    static_assert(
        !is_int_type(parameter_type_param) || std::is_same_v<value_type, int>, "The given data must be of type int"
    );
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
        kassert_not_extracted("Cannot get the size of a buffer that has already been extracted.");
        if constexpr (is_single_element) {
            return 1;
        } else {
            return underlying().size();
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
        kassert_not_extracted("Cannot resize a buffer that has already been extracted.");
        if constexpr (is_single_element) {
            KASSERT(
                size == 1u,
                "Cannot resize a single element buffer to hold zero or more than one element. Single "
                "element buffers always hold exactly one element."
            );
        } else if constexpr (std::is_same_v<MemberType, Span<value_type>>) {
            KASSERT(this->size() >= size, "Span cannot be resized and is smaller than the requested size.");
        } else {
            underlying().resize(size);
        }
    }

    /// @brief Get const access to the underlying container.
    /// @return Pointer to the underlying container.
    value_type const* data() const {
        kassert_not_extracted("Cannot get a pointer to a buffer that has already been extracted.");
        if constexpr (is_single_element) {
            return &underlying();
        } else {
            return std::data(underlying());
        }
    }

    /// @brief Get access to the underlying container.
    /// @return Pointer to the underlying container.
    value_type_with_const* data() {
        kassert_not_extracted("Cannot get a pointer to a buffer that has already been extracted.");
        if constexpr (is_single_element) {
            return &underlying();
        } else {
            return std::data(underlying());
        }
    }

    /// @brief Get read-only access to the underlying storage.
    /// @return Span referring the underlying storage.
    Span<value_type const> get() const {
        kassert_not_extracted("Cannot get a buffer that has already been extracted.");
        return {this->data(), this->size()};
    }

    /// @brief Get access to the underlying storage.
    /// @return Span referring to the underlying storage.
    Span<value_type_with_const> get() {
        kassert_not_extracted("Cannot get a buffer that has already been extracted.");
        return {this->data(), this->size()};
    }

    /// @brief Get the single element wrapped by this object.
    /// @return The single element wrapped by this object.
    template <bool enabled = is_single_element, std::enable_if_t<enabled, bool> = true>
    value_type const get_single_element() const {
        kassert_not_extracted("Cannot get an element from a buffer that has already been extracted.");
        return underlying();
    }

    /// @brief Provides access to the underlying data.
    /// @return A reference to the data.
    MemberType const& underlying() const {
        kassert_not_extracted("Cannot get a buffer that has already been extracted.");
        // this assertion is only checked if the buffer is actually accessed.
        static_assert(
            !is_vector_bool_v<MemberType>,
            "Buffers based on std::vector<bool> are not supported, use std::vector<kamping::kabool> instead."
        );
        return _data;
    }

    /// @brief Provides access to the underlying data.
    /// @return A reference to the data.
    template <bool enabled = modifiability == BufferModifiability::modifiable, std::enable_if_t<enabled, bool> = true>
    MemberType& underlying() {
        kassert_not_extracted("Cannot get a buffer that has already been extracted.");
        // this assertion is only checked if the buffer is actually accessed.
        static_assert(
            !is_vector_bool_v<MemberType>,
            "Buffers based on std::vector<bool> are not supported, use std::vector<kamping::kabool> instead."
        );
        return _data;
    }

    /// @brief Extract the underlying container. This will leave the DataBuffer in an unspecified
    /// state.
    ///
    /// @return Moves the underlying container out of the DataBuffer.
    template <bool enabled = allocation == BufferAllocation::lib_allocated, std::enable_if_t<enabled, bool> = true>
    MemberTypeWithConst extract() {
        static_assert(
            ownership == BufferOwnership::owning,
            "Moving out of a reference should not be done because it would leave "
            "a users container in an unspecified state."
        );
        kassert_not_extracted("Cannot extract a buffer that has already been extracted.");
        auto extracted = std::move(underlying());
        // we set is_extracted here because otherwise the call to underlying() would fail
        set_extracted();
        return extracted;
    }

private:
    /// @brief Set the extracted flag to indicate that the data stored in this buffer has been moved out.
    void set_extracted() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        is_extracted = true;
#endif
    }

    /// @brief Throws an assertion if the extracted flag is set, i.e. the underlying data has been moved out.
    ///
    /// @param message The message for the assertion.
    void kassert_not_extracted(std::string const message [[maybe_unused]]) const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, message, assert::normal);
#endif
    }

    MemberTypeWithConstAndRef _data; ///< Container which holds the actual data.
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    bool is_extracted = false; ///< Has the container been extracted and is therefore in an invalid state?
#endif
};

/// @brief Empty buffer that can be used as default argument for optional buffer parameters.
/// @tparam ParameterType Parameter type represented by this pseudo buffer.
template <typename Data, ParameterType type>
class EmptyDataBuffer {
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

/// @brief Encapsulates the rank of a PE. This is needed for p2p communicaiton and rooted \c MPI collectives like \c
/// MPI_Gather.
///
/// This is a specialized \c DataBuffer. Its main functionality is to provide ease-of-use functionality in the form of
/// the methods \c rank() and \c rank_signed(), which return the ecapsulated rank and are easier to read in the code.
template <ParameterType type>
class RankDataBuffer final : public DataBuffer<
                                 size_t,
                                 ParameterType::root,
                                 BufferModifiability::modifiable,
                                 BufferOwnership::owning,
                                 BufferType::in_buffer,
                                 BufferAllocation::user_allocated> {
public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this object encapsulates.

    /// @ Constructor for Rank.
    /// @param rank Rank of the PE.
    RankDataBuffer(size_t rank) : DataBuffer(rank) {}

    /// @ Constructor for Rank.
    /// @param rank Rank of the PE.
    RankDataBuffer(int rank) : DataBuffer(asserting_cast<size_t>(rank)) {}

    /// @brief Returns the rank as `size_t`.
    /// @returns Rank of the PE as `size_t`.
    size_t rank() const {
        return underlying();
    }

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return asserting_cast<int>(rank());
    }
};

using RootDataBuffer = RankDataBuffer<ParameterType::root>; ///< Helper for roots;

} // namespace internal

/// @}

} // namespace kamping
