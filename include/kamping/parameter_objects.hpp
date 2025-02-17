// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.
/// @file
/// @brief Parameter objects return by named parameter factory functions

#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/serialization.hpp"
#include "kamping/status.hpp"

namespace kamping::internal {

/// @brief Dummy template for representing the absence of a container to rebind to.
/// @see AllocNewDataBufferBuilder::construct_buffer_or_rebind()
template <typename>
struct UnusedRebindContainer {};

/// @brief Parameter object representing a data buffer. This is an intermediate object which only holds the data and
/// parameters. The actual buffer is created by calling the \c construct_buffer_or_rebind() method.
/// @tparam Data The data type.
/// @tparam parameter_type_param The parameter type.
/// @tparam modifiability The modifiability of the buffer.
/// @tparam buffer_type The type of the buffer.
/// @tparam buffer_resize_policy The resize policy of the buffer.
/// @tparam ValueType The value type of the buffer. Defaults to \ref default_value_type_tag, indicating that this buffer
/// does not enforce a specific value type.
template <
    typename Data,
    ParameterType       parameter_type_param,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename ValueType = default_value_type_tag>
struct DataBufferBuilder {
    static constexpr ParameterType parameter_type = parameter_type_param; ///< The parameter type.
    DataBufferBuilder() : data_() {}
    /// @brief Constructor for DataBufferBuilder.
    /// @param data The container to build a databuffer for
    /// @tparam Data_ The type of the container.
    template <typename Data_>
    DataBufferBuilder(Data_&& data) : data_(std::forward<Data_>(data)) {}

private:
    Data data_;

public:
    using DataBufferType =
        decltype(make_data_buffer<
                 ParameterType,
                 parameter_type,
                 modifiability,
                 buffer_type,
                 buffer_resize_policy,
                 ValueType>(std::forward<Data>(data_))); ///< The type of the constructed data buffer.

    /// @brief Constructs the data buffer.
    /// @tparam RebindContainerType The container to use for the data buffer (has no effect here).
    /// @tparam Flag A tag type indicating special behavior, e.g., serialization support (@see \ref
    /// serialization_support_tag). Defaults to `void`.
    template <template <typename...> typename RebindContainerType = UnusedRebindContainer, typename Flag = void>
    auto construct_buffer_or_rebind() {
        using Data_no_ref                           = std::remove_const_t<std::remove_reference_t<Data>>;
        static constexpr bool support_serialization = std::is_same_v<Flag, internal::serialization_support_tag>;
        static_assert(
            support_serialization || !internal::is_serialization_buffer_v<Data_no_ref>,
            "\n ---> Serialization buffers are not supported here."
        );
        if constexpr (is_empty_data_buffer_v<Data_no_ref>) {
            return internal::EmptyDataBuffer<ValueType, parameter_type, buffer_type>{};
        } else {
            return make_data_buffer<
                ParameterType,
                parameter_type,
                modifiability,
                buffer_type,
                buffer_resize_policy,
                ValueType>(std::forward<Data>(data_));
        }
    }
    static constexpr bool is_out_buffer =
        DataBufferType::is_out_buffer; ///< \c true if the buffer is an out or in/out buffer that results will be
                                       ///< written to and \c false otherwise.
    static constexpr bool is_owning =
        DataBufferType::is_owning; ///< Indicates whether the buffer owns its underlying storage.
    static constexpr bool is_lib_allocated =
        DataBufferType::is_lib_allocated; ///< Indicates whether the buffer is allocated by KaMPIng.
    static constexpr bool is_single_element =
        DataBufferType::is_single_element; ///< Indicated whether the buffer is a single element buffer.
    using value_type = typename DataBufferType::value_type; ///< The constructed data buffer's value type.

    /// @brief The size of the underlying container.
    size_t size() const {
        if constexpr (is_single_element) {
            return 1;
        } else {
            return data_.size();
        }
    }
};

/// @brief Parameter object representing a data buffer to be allocated by KaMPIng. This is a specialization of \ref
/// DataBufferBuilder for buffer allocation tags, such as \ref alloc_new, \ref alloc_new_using and \ref
/// alloc_container_of. This is an intermediate object not holding any data. The actual buffer is constructed by
/// calling the \c construct_buffer_or_rebind() method.
///
/// This type should be constructed using the factory methods \ref make_data_buffer_builder.
///
/// @tparam AllocType A tag type indicating what kind of buffer should be allocated. see \ref alloc_new, \ref
/// alloc_new_using and \ref alloc_container_of.
template <
    typename AllocType,
    typename ValueType,
    ParameterType       parameter_type_param,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy>
struct AllocNewDataBufferBuilder {
    static constexpr ParameterType parameter_type = parameter_type_param; ///< The parameter type.
public:
    using DataBufferType = decltype(make_data_buffer<
                                    ParameterType,
                                    parameter_type,
                                    modifiability,
                                    buffer_type,
                                    buffer_resize_policy,
                                    ValueType>(
        std::conditional_t<
            is_alloc_container_of_v<AllocType>,
            AllocNewT<std::vector<ValueType>>, // we rebind to std::vector here, because this DataBufferType is only
                                               // used for determining is_out_buffer, is_owning, etc. and rebinding
                                               // does not affect this.
            AllocType>{}
    )); ///< The type of the constructed data buffer (potentially rebinded to std::vector).

public:
    /// @brief Constructs the data buffer.
    /// @tparam RebindContainerType The container to use for constructing the data buffer. This parameter is ignored if
    /// the buffer allocation trait is \ref alloc_new or \ref alloc_new_using. In case of `alloc_container_of<U>`, the
    /// created data buffer encapsulated a `RebindContainerType<U>`.
    /// @tparam Flag A tag type indicating special behavior, e.g., serialization support (@see \ref
    /// serialization_support_tag). Defaults to `void`.
    template <template <typename...> typename RebindContainerType = UnusedRebindContainer, typename Flag = void>
    auto construct_buffer_or_rebind() {
        if constexpr (is_alloc_new_v<AllocType>) {
            return make_data_buffer<
                ParameterType,
                parameter_type,
                modifiability,
                buffer_type,
                buffer_resize_policy,
                ValueType>(alloc_new<typename AllocType::container_type>);
        } else if constexpr (is_alloc_new_using_v<AllocType>) {
            return make_data_buffer<
                ParameterType,
                parameter_type,
                modifiability,
                buffer_type,
                buffer_resize_policy,
                ValueType>(alloc_new_using<AllocType::template container_type>);
        } else if constexpr (is_alloc_container_of_v<AllocType>) {
            static_assert(
                !std::is_same_v<RebindContainerType<void>, UnusedRebindContainer<void>>,
                "RebindContainerType is required."
            );
            return make_data_buffer<
                ParameterType,
                parameter_type,
                modifiability,
                buffer_type,
                buffer_resize_policy,
                ValueType>(alloc_new<RebindContainerType<typename AllocType::value_type>>);
        } else {
            static_assert(is_alloc_container_of_v<AllocType>, "Unknown AllocType");
        }
    }
    static constexpr bool is_out_buffer =
        DataBufferType::is_out_buffer; ///< \c true if the buffer is an out or in/out buffer that results will be
                                       ///< written to and \c false otherwise.
    static constexpr bool is_owning =
        DataBufferType::is_owning; ///< Indicates whether the buffer owns its underlying storage.
    static constexpr bool is_lib_allocated =
        DataBufferType::is_lib_allocated; ///< Indicates whether the buffer is allocated by KaMPIng
    static constexpr bool is_single_element =
        DataBufferType::is_single_element; ///< Indicated whether the buffer is a single element buffer.
    using value_type = typename DataBufferType::value_type; ///< The constructed data buffer's value type.

    /// @brief The size of the underlying container.
    size_t size() const {
        if constexpr (is_single_element) {
            return 1;
        } else {
            return 0;
        }
    }
};

/// @brief Factory method for constructing a \ref DataBufferBuilder from the given Container \p Data.
/// @see DataBufferBuilder
template <
    ParameterType       parameter_type,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename ValueType = default_value_type_tag,
    typename Data>
auto make_data_buffer_builder(Data&& data) {
    return DataBufferBuilder<Data, parameter_type, modifiability, buffer_type, buffer_resize_policy, ValueType>(
        std::forward<Data>(data)
    );
}

/// @brief Factory method for constructing a \ref DataBufferBuilder from an `std::initializer_list`.
/// @see DataBufferBuilder
template <
    ParameterType       parameter_type,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename Data>
auto make_data_buffer_builder(std::initializer_list<Data> data) {
    auto data_vec = [&]() {
        if constexpr (std::is_same_v<Data, bool>) {
            return std::vector<kabool>(data.begin(), data.end());
            // We only use automatic conversion of bool to kabool for initializer lists, but not for single elements of
            // type bool. The reason for that is, that sometimes single element conversion may not be desired.
            // E.g. consider a gather operation with send_buf := bool& and recv_buf := Span<bool>, or a bcast with
            // send_recv_buf = bool&
        } else {
            return std::vector<Data>{data};
        }
    }();
    return DataBufferBuilder<decltype(data_vec), parameter_type, modifiability, buffer_type, buffer_resize_policy>(
        std::move(data_vec)
    );
}

/// @brief Factory method for constructing an \ref AllocNewDataBufferBuilder for \ref alloc_new.
template <
    ParameterType       parameter_type,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename ValueType = default_value_type_tag,
    typename Data>
auto make_data_buffer_builder(AllocNewT<Data>) {
    return AllocNewDataBufferBuilder<
        AllocNewT<Data>,
        ValueType,
        parameter_type,
        modifiability,
        buffer_type,
        buffer_resize_policy>();
}

/// @brief Factory method for constructing an \ref AllocNewDataBufferBuilder for \ref alloc_new_using.
template <
    ParameterType       parameter_type,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename ValueType = default_value_type_tag,
    template <typename...>
    typename Container>
auto make_data_buffer_builder(AllocNewUsingT<Container>) {
    return AllocNewDataBufferBuilder<
        AllocNewUsingT<Container>,
        ValueType,
        parameter_type,
        modifiability,
        buffer_type,
        buffer_resize_policy>();
}

/// @brief Factory method for constructing an \ref AllocNewDataBufferBuilder for \ref alloc_container_of.
template <
    ParameterType       parameter_type,
    BufferModifiability modifiability,
    BufferType          buffer_type,
    BufferResizePolicy  buffer_resize_policy,
    typename ValueType>
auto make_data_buffer_builder(AllocContainerOfT<ValueType>) {
    return AllocNewDataBufferBuilder<
        AllocContainerOfT<ValueType>,
        ValueType,
        parameter_type,
        modifiability,
        buffer_type,
        buffer_resize_policy>();
}

/// @brief Factory method for constructing an DataBufferBuilder for an \ref EmptyDataBuffer.
/// @see DataBufferBuilder
template <typename ValueType, ParameterType parameter_type, BufferType buffer_type>
auto make_empty_data_buffer_builder() {
    return DataBufferBuilder<
        EmptyDataBuffer<ValueType, parameter_type, buffer_type>,
        parameter_type,
        BufferModifiability::constant,
        buffer_type,
        no_resize,
        ValueType>();
}

/// @brief Helper type for representing a type list
/// @tparam Args the types.
template <typename... Args>
struct type_list {
    /// @brief Member attribute to check if a type is contained in the list
    /// @tparam T The type to check for if it is contained in the list.
    template <typename T>
    static constexpr bool contains = std::disjunction<std::is_same<T, Args>...>::value;
};

/// @brief Tag type for parameters that can be omitted on some PEs (e.g., root
/// PE, or non-root PEs).
template <typename T>
struct ignore_t {};

/// @brief Indicator if a rank parameter holds and actual value or \c MPI_ANY_SOURCE or \c MPI_PROC_NULL.
enum class RankType {
    value, ///< holds a value
    any,   ///< holds \c MPI_ANY_SOURCE
    null   ///< holds \c MPI_PROC_NULL
};

struct rank_any_t {};  ///< tag struct for \c MPI_ANY_SOURCE
struct rank_null_t {}; ///< tag struct for \c MPI_PROC_NULL

/// @brief Encapsulates the rank of a PE. This is needed for p2p communication
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialized \c DataBuffer. Its main functionality is to provide
/// ease-of-use functionality in the form of the methods \c rank() and \c
/// rank_signed(), which return the encapsulated rank and are easier to read in
/// the code.
// @tparam rank_type The \ref RankType encapsulated.
// @tparam parameter_type The parameter type.
template <RankType rank_type, ParameterType parameter_type>
class RankDataBuffer {};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communication
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialized \c DataBuffer. Its main functionality is to provide
/// ease-of-use functionality in the form of the methods \c rank() and \c
/// rank_signed(), which return the encapsulated rank and are easier to read in
/// the code.
// @tparam rank_type The \ref RankType encapsulated.
// @tparam parameter_type The parameter type.
template <ParameterType type>
class RankDataBuffer<RankType::value, type> final : private DataBuffer<
                                                        size_t,
                                                        ParameterType,
                                                        type,
                                                        BufferModifiability::modifiable,
                                                        BufferOwnership::owning,
                                                        BufferType::in_buffer,
                                                        BufferResizePolicy::no_resize,
                                                        BufferAllocation::user_allocated> {
private:
    using BaseClass = DataBuffer<
        size_t,
        ParameterType,
        type,
        BufferModifiability::modifiable,
        BufferOwnership::owning,
        BufferType::in_buffer,
        BufferResizePolicy::no_resize,
        BufferAllocation::user_allocated>;

public:
    static constexpr ParameterType parameter_type = type; ///< The type of parameter this object encapsulates.
    static constexpr RankType      rank_type      = RankType::value; ///< The rank type.

    /// @brief Constructor for Rank.
    /// @param rank Rank of the PE.
    RankDataBuffer(size_t rank) : BaseClass(rank) {}

    /// @brief Constructor for Rank.
    /// @param rank Rank of the PE.
    RankDataBuffer(int rank) : BaseClass(asserting_cast<size_t>(rank)) {}

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return asserting_cast<int>(BaseClass::underlying());
    }

    /// @brief Get a copy of this RankDataBuffer.
    ///
    /// @return A copy of this RankDataBuffer.
    RankDataBuffer<rank_type, parameter_type> clone() {
        return {BaseClass::underlying()};
    }
};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communication
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialization for MPI_ANY_SOURCE which only implements
/// \ref rank_signed(), without allocating any additional memory.
template <ParameterType type>
class RankDataBuffer<RankType::any, type> : private CopyMoveEnabler<> {
public:
    static constexpr ParameterType parameter_type = type;          ///< The type of parameter this object encapsulates.
    static constexpr RankType      rank_type      = RankType::any; ///< The rank type.

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return MPI_ANY_SOURCE;
    }

    /// @brief Get a copy of this RankDataBuffer.
    ///
    /// @return A copy of this RankDataBuffer.
    RankDataBuffer<rank_type, parameter_type> clone() {
        return {};
    }
};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communication
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialization for MPI_PROC_NULL which only implements
/// \ref rank_signed(), without allocating any additional memory.
template <ParameterType type>
class RankDataBuffer<RankType::null, type> : private CopyMoveEnabler<> {
public:
    static constexpr ParameterType parameter_type = type;           ///< The type of parameter this object encapsulates.
    static constexpr RankType      rank_type      = RankType::null; ///< The rank type.

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return MPI_PROC_NULL;
    }

    /// @brief Get a copy of this RankDataBuffer.
    ///
    /// @return A copy of this RankDataBuffer.
    RankDataBuffer<rank_type, parameter_type> clone() {
        return {};
    }
};

using RootDataBuffer = RankDataBuffer<RankType::value, ParameterType::root>; ///< Helper for roots;

struct standard_mode_t {};    ///< tag for standard send mode
struct buffered_mode_t {};    ///< tag for buffered send mode
struct synchronous_mode_t {}; ///< tag for synchronous send mode
struct ready_mode_t {};       ///< tag for ready send mode
using send_mode_list =
    type_list<standard_mode_t, buffered_mode_t, synchronous_mode_t, ready_mode_t>; ///< list of all available send modes

/// @brief Parameter object for send_mode encapsulating the send mode compile-time tag.
/// @tparam SendModeTag The send mode.
template <typename SendModeTag>
struct SendModeParameter : private CopyMoveEnabler<> {
    static_assert(send_mode_list::contains<SendModeTag>, "Unsupported send mode.");
    static constexpr ParameterType parameter_type = ParameterType::send_mode; ///< The parameter type.
    using send_mode                               = SendModeTag;              ///< The send mode.
};

/// @brief returns a pointer to the \c MPI_Status encapsulated by the provided status parameter object.
/// @tparam StatusParam The type of the status parameter object.
/// @param param The status parameter object.
/// @returns A pointer to the encapsulated \c MPI_Status or \c MPI_STATUS_IGNORE.
template <typename StatusParam>
static inline MPI_Status* status_param_to_native_ptr(StatusParam& param) {
    static_assert(StatusParam::parameter_type == ParameterType::status);
    static_assert(type_list<MPI_Status, Status>::contains<typename StatusParam::value_type>);
    if constexpr (StatusParam::buffer_type == BufferType::ignore) {
        return MPI_STATUS_IGNORE;
    } else if constexpr (std::is_same_v<typename StatusParam::value_type, MPI_Status>) {
        return param.data();
    } else {
        // value_type == kamping::Status
        return &param.underlying().native();
    }
}

struct any_tag_t {}; ///< tag struct for message tag

/// @brief Possible types of tag
enum class TagType {
    value, ///< holds an actual value
    any    ///< special value MPI_ANY_TAG}
};

/// @brief Encapsulates a message tag.
/// @tparam The type of the tag.
/// @tparam The parameter type associated with the tag parameter object. Defaults to \ref ParameterType::tag.
template <TagType tag_type, ParameterType parameter_type = ParameterType::tag>
class TagParam {};

/// @brief Encapsulates a message tag. Specialization if an explicit tag value is provided.
template <ParameterType parameter_type_>
class TagParam<TagType::value, parameter_type_> : private CopyMoveEnabler<> {
public:
    /// @param tag The tag.
    TagParam(int tag) : _tag_value(tag) {}
    static constexpr ParameterType parameter_type = parameter_type_; ///< The parameter type.
    static constexpr TagType       tag_type       = TagType::value;  ///< The tag type.
    /// @return The tag.
    [[nodiscard]] int tag() const {
        return _tag_value;
    }

    /// @brief Get a copy of this TagParam.
    ///
    /// @return A copy of this TagParam.
    TagParam<tag_type> clone() {
        return {_tag_value};
    }

private:
    int _tag_value; ///< the encapsulated tag value
};

/// @brief Encapsulates a message tag. Specialization if the value is MPI_ANY_TAG.
template <ParameterType parameter_type_>
class TagParam<TagType::any, parameter_type_> : private CopyMoveEnabler<> {
public:
    static constexpr ParameterType parameter_type = parameter_type_; ///< The parameter type.
    static constexpr TagType       tag_type       = TagType::any;    ///< The tag type.
    /// @return The tag.
    [[nodiscard]] int tag() const {
        return MPI_ANY_TAG;
    }

    /// @brief Get a copy of this TagParam.
    ///
    /// @return A copy of this TagParam.
    TagParam<tag_type> clone() {
        return {};
    }
};
} // namespace kamping::internal

namespace kamping {

namespace send_modes {
static constexpr internal::standard_mode_t    standard{};    ///< global constant for standard send mode
static constexpr internal::buffered_mode_t    buffered{};    ///< global constant for buffered send mode
static constexpr internal::synchronous_mode_t synchronous{}; ///< global constant for synchronous send mode
static constexpr internal::ready_mode_t       ready{};       ///< global constant for ready send mode
} // namespace send_modes

/// @brief Tag for parameters that can be omitted on some PEs (e.g., root PE, or non-root PEs).
template <typename T = void>
constexpr internal::ignore_t<T> ignore{};

namespace tags {
static constexpr internal::any_tag_t any{}; ///< global constant for any tag
}

namespace rank {
static constexpr internal::rank_any_t  any{};  ///< global constant for any rank
static constexpr internal::rank_null_t null{}; ///< global constant for rank NULL
} // namespace rank

} // namespace kamping
