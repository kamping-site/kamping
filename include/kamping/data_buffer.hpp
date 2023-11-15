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
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/span.hpp"
#include "kassert/kassert.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {

/// @brief Base object for parameter objects which deletes copy constructor and assignment operator and enables move.
///
/// You can inherit from this class privately.
/// While  constructors are never inherited, the derived class still has no copy constructor (assignment), because it
/// can not be default constructed, due to the missing implementation in the base class. Because we provide a (default)
/// implementation for the move constructor (assignment) in the base class, the derived class can construct default
/// implementations.
class ParameterObjectBase {
protected:
    constexpr ParameterObjectBase() = default;
    ~ParameterObjectBase()          = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    ParameterObjectBase(ParameterObjectBase const&) = delete;
    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    ParameterObjectBase& operator=(ParameterObjectBase const&) = delete;
    /// @brief Move constructor.
    ParameterObjectBase(ParameterObjectBase&&) = default;
    /// @brief Move assignment operator.
    ParameterObjectBase& operator=(ParameterObjectBase&&) = default;

    /// @brief Set the extracted flag to indicate that the status stored in this buffer has been moved out.
    void set_extracted() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        is_extracted = true;
#endif
    }

    /// @brief Throws an assertion if the extracted flag is set, i.e. the underlying status has been moved out.
    ///
    /// @param message The message for the assertion.
    void kassert_not_extracted(std::string const message [[maybe_unused]]) const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, message, assert::normal);
#endif
    }

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    bool is_extracted = false; ///< Has the status been extracted and is therefore in an invalid state?
#endif
};

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

/// @brief Type used for indicating that a buffer should be allocated by KaMPIng.
/// @tparam Container The container to allocate.
///
/// Passing this with an appropriate template parameter to a buffer creation function (such as \c recv_buf()) indicates,
/// that the MPI operation should allocate an appropriately sized buffer of type \c Container internally.
template <typename Container>
struct AllocNewT {};

/// @brief Convenience wrapper for creating library allocated containers. See \ref AllocNewT for details.
template <typename Container>
static constexpr auto alloc_new = AllocNewT<Container>{};

/// @brief Type used for indicating that a buffer should be allocated by KaMPIng.
/// @tparam Container A container template to use for allocation.
///
/// Passing this with an appropriate template parameter to a buffer creation function (such as \c recv_counts())
/// indicates, that the MPI operation should allocate an appropriately sized buffer of type \c Container<T> internally,
/// where \c T is automatically determined.
///
/// In case of \c recv_counts(alloc_new_auto<std::vector>) this means, that internally, a \c std::vector<int> is
/// allocated.
template <template <typename...> typename Container>
struct AllocNewAutoT {};

/// @brief Convenience wrapper for creating library allocated containers. See \ref AllocNewAutoT for details.
template <template <typename...> typename Container>
static constexpr auto alloc_new_auto = AllocNewAutoT<Container>{};

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
enum class BufferType { in_buffer, out_buffer, in_out_buffer, ignore };
} // namespace internal

/// @brief Enum to specify in which cases a buffer is resized.
enum class BufferResizePolicy {
    no_resize,    ///< Policy indicating that the underlying buffer shall never be resized.
    grow_only,    ///< Policy indicating that the underlying buffer shall only be resized if the current size
                  ///< of the buffer is too small.
    resize_to_fit ///< Policy indicating that the underlying buffer is resized such that it has exactly the required
                  ///< size.
};

constexpr BufferResizePolicy no_resize =
    BufferResizePolicy::no_resize; ///< Constant storing a BufferResizePolicy::no_resize enum member. It can be used to
                                   ///< declare a buffer's resize policy in more concise manner.
constexpr BufferResizePolicy grow_only =
    BufferResizePolicy::grow_only; ///< Constant storing a BufferResizePolicy::grow_only enum member. It can be used to
                                   ///< declare a buffer's resize policy in more concise manner.
constexpr BufferResizePolicy resize_to_fit =
    BufferResizePolicy::resize_to_fit; ///< Constant storing a BufferResizePolicy::resize_to_fit enum member. It can be
                                       ///< used to declare a buffer's resize policy in more concise manner.

namespace internal {
/// @brief Wrapper to get the value type of a non-container type (aka the type itself).
/// @tparam has_value_type_member Whether `T` has a value_type member
/// @tparam T The type to get the value_type of
template <bool has_value_type_member /*= false */, typename T>
class ValueTypeWrapper {
public:
    using value_type = T; ///< The value type of T.
};

/// @brief tag type to indicate that the value_type should be inferred from the container
struct default_value_type_tag {};

/// @brief Wrapper to get the value type of a container type.
/// @tparam T The type to get the value_type of
template <typename T>
class ValueTypeWrapper</*has_value_type_member =*/true, T> {
public:
    using value_type = typename T::value_type; ///< The value type of T.
};

/// @brief for a given \param MemberType of a data buffer, defines the most viable resize policy.
///
/// For example, a single element buffer may not be resizable.
template <typename MemberType>
constexpr BufferResizePolicy maximum_viable_resize_policy = [] {
    auto is_single_element = !has_data_member_v<MemberType>;
    if (is_single_element) {
        return no_resize;
    } else {
        return resize_to_fit;
    }
}();

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
/// @tparam buffer_resize_policy_param Policy specifying whether (and if so, how) the underlying buffer shall be
/// resized.
/// @tparam allocation `lib_allocated` if the buffer was allocated by the library,
/// @tparam ValueType requested value_type for the buffer. If it does not match the containers value type, compilation
/// fails. By default, this is set to \c default_value_type_tag and the value_type is inferred from the underlying
/// container, without any checking `user_allocated` if it was allocated by the user.
template <
    typename MemberType,
    ParameterType       parameter_type_param,
    BufferModifiability modifiability,
    BufferOwnership     ownership,
    BufferType          buffer_type_param,
    BufferResizePolicy  buffer_resize_policy_param,
    BufferAllocation    allocation = BufferAllocation::user_allocated,
    typename ValueType             = default_value_type_tag>
class DataBuffer : private ParameterObjectBase {
public:
    static constexpr ParameterType parameter_type =
        parameter_type_param; ///< The type of parameter this buffer represents.

    static constexpr BufferType buffer_type = buffer_type_param; ///< The type of the buffer, i.e., in, out, or in_out.

    static constexpr BufferResizePolicy resize_policy =
        buffer_resize_policy_param; ///< The policy specifying in which cases the buffer shall be resized.

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
    static_assert(
        std::is_same_v<ValueType, default_value_type_tag> || std::is_same_v<ValueType, value_type>,
        "The requested value type of the buffer does not match the value type of the underlying container"
    );
    using value_type_with_const =
        std::conditional_t<is_modifiable, value_type, value_type const>; ///< Value type as const or non-const depending
                                                                         ///< on modifiability
    static_assert(
        is_modifiable || resize_policy == BufferResizePolicy::no_resize,
        "A constant data buffer requires the that the resize policy is no_resize."
    );
    static_assert(
        !is_single_element || resize_policy == BufferResizePolicy::no_resize,
        "A single element data buffer requires the that the resize policy is no_resize."
    );

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

    /// @brief The size of the underlying container.
    size_t size() const {
        kassert_not_extracted("Cannot get the size of a buffer that has already been extracted.");
        if constexpr (is_single_element) {
            return 1;
        } else {
            return underlying().size();
        }
    }

    /// @brief Resizes the underlying container such that it holds exactly \c size elements of \c value_type.
    ///
    /// This function calls \c resize on the underlying container.
    ///
    /// This takes only part in overload resolution if the \ref resize_policy of the buffer is \ref resize_to_fit.
    ///
    /// @param size Size the container is resized to.
    template <
        BufferResizePolicy _resize_policy                                = resize_policy,
        typename std::enable_if_t<_resize_policy == resize_to_fit, bool> = true>
    void resize(size_t size) {
        kassert_not_extracted("Cannot resize a buffer that has already been extracted.");
        underlying().resize(size);
    }

    /// @brief Resizes the underlying container such that it holds at least \c size elements of \c value_type.
    ///
    /// This function calls \c resize on the underlying container, but only if the requested \param size is larger than
    /// the current buffer size. Otherwise, the buffer is left unchanged.
    ///
    /// This takes only part in overload resolution if the \ref resize_policy of the buffer is \ref grow_only.
    ///
    /// @param size Size the container is resized to.
    template <
        BufferResizePolicy _resize_policy                            = resize_policy,
        typename std::enable_if_t<_resize_policy == grow_only, bool> = true>
    void resize(size_t size) {
        kassert_not_extracted("Cannot resize a buffer that has already been extracted.");
        if (this->size() < size) {
            underlying().resize(size);
        }
    }

    template <
        BufferResizePolicy _resize_policy                            = resize_policy,
        typename std::enable_if_t<_resize_policy == no_resize, bool> = true>
    void resize(size_t size) = delete;

    /// @brief Resizes the underlying container if the buffer the buffer's resize policy allows and resizing is
    /// necessary.
    ///
    /// @tparam SizeFunc Type of the functor which computes the required buffer size.
    /// @param compute_required_size Functor which is used to compute the required buffer size. compute_required_size()
    /// is not called if the buffer's resize policy is BufferResizePolicy::no_resize.
    template <typename SizeFunc>
    void resize_if_requested(SizeFunc&& compute_required_size) {
        if constexpr (resize_policy == BufferResizePolicy::resize_to_fit || resize_policy == BufferResizePolicy::grow_only) {
            resize(compute_required_size());
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
    MemberTypeWithConstAndRef _data; ///< Container which holds the actual data.
};

/// @brief Empty buffer that can be used as default argument for optional buffer parameters.
/// @tparam ParameterType Parameter type represented by this pseudo buffer.
template <typename Data, ParameterType type, BufferType buffer_type_param>
class EmptyDataBuffer {
public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this buffer represents.
    static constexpr bool          is_modifiable =
        false;               ///< This pseudo buffer is not modifiable since it represents no actual buffer.
    using value_type = Data; ///< Value type of the buffer.
    static constexpr BufferType buffer_type =
        buffer_type_param; ///< The type of the buffer, usually ignore for this special buffer.

    static constexpr BufferResizePolicy resize_policy     = no_resize; ///< An empty buffer can not be resized.
    static constexpr bool               is_out_buffer     = false;     ///< An empty buffer is never output.
    static constexpr bool               is_lib_allocated  = false;     ///< An empty buffer is not allocated.
    static constexpr bool               is_single_element = false;     ///< An empty contains no elements.

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
    /// @brief Resizes the underlying container if the buffer the buffer's resize policy allows and resizing is
    /// necessary. Does nothing for an empty buffer.
    ///
    /// @tparam SizeFunc Type of the functor which computes the required buffer size.
    /// @param compute_required_size Functor which is used to compute the required buffer size. compute_required_size()
    /// is not called if the buffer's resize policy is BufferResizePolicy::no_resize.
    template <typename SizeFunc>
    void resize_if_requested(SizeFunc&& compute_required_size [[maybe_unused]]) {}
};

} // namespace internal

/// @}

} // namespace kamping
