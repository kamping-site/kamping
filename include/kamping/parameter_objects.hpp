// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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

namespace kamping::internal {

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

/// @brief Indicator if a rank parameter holds and actual value or MPI_ANY_SOURCE or MPI_PROC_NULL.
enum class RankType {
    value, ///< holds a value
    any,   ///< holds MPI_ANY_SOURCE
    null   ///< hold MPI_PROC_NULL
};

struct rank_any_t {};  ///< tag struct for MPI_ANY_SOURCE
struct rank_null_t {}; ///< tag struct for MPI_PROC_NULL

/// @brief Encapsulates the rank of a PE. This is needed for p2p communicaiton
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialized \c DataBuffer. Its main functionality is to provide
/// ease-of-use functionality in the form of the methods \c rank() and \c
/// rank_signed(), which return the ecapsulated rank and are easier to read in
/// the code.
// @tparam rank_type The \ref RankType encapsulated.
// @tparam parameter_type The parameter type.
template <RankType rank_type, ParameterType parameter_type>
class RankDataBuffer {};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communicaiton
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialized \c DataBuffer. Its main functionality is to provide
/// ease-of-use functionality in the form of the methods \c rank() and \c
/// rank_signed(), which return the ecapsulated rank and are easier to read in
/// the code.
// @tparam rank_type The \ref RankType encapsulated.
// @tparam parameter_type The parameter type.
template <ParameterType type>
class RankDataBuffer<RankType::value, type> final : private DataBuffer<
                                                        size_t,
                                                        type,
                                                        BufferModifiability::modifiable,
                                                        BufferOwnership::owning,
                                                        BufferType::in_buffer,
                                                        BufferAllocation::user_allocated> {
private:
    using BaseClass = DataBuffer<
        size_t,
        type,
        BufferModifiability::modifiable,
        BufferOwnership::owning,
        BufferType::in_buffer,
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
};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communicaiton
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialization for MPI_ANY_SOURCE which only implements
/// \ref rank_signed(), without allocating any additional memory.
template <ParameterType type>
class RankDataBuffer<RankType::any, type> : private ParameterObjectBase {
public:
    static constexpr ParameterType parameter_type = type;          ///< The type of parameter this object encapsulates.
    static constexpr RankType      rank_type      = RankType::any; ///< The rank type.

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return MPI_ANY_SOURCE;
    }
};

/// @brief Encapsulates the rank of a PE. This is needed for p2p communicaiton
/// and rooted \c MPI collectives like \c MPI_Gather.
///
/// This is a specialization for MPI_PROC_NULL which only implements
/// \ref rank_signed(), without allocating any additional memory.
template <ParameterType type>
class RankDataBuffer<RankType::null, type> : private ParameterObjectBase {
public:
    static constexpr ParameterType parameter_type = type;           ///< The type of parameter this object encapsulates.
    static constexpr RankType      rank_type      = RankType::null; ///< The rank type.

    /// @brief Returns the rank as `int`.
    /// @returns Rank as `int`.
    int rank_signed() const {
        return MPI_PROC_NULL;
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
struct SendModeParameter : private ParameterObjectBase {
    static_assert(send_mode_list::contains<SendModeTag>, "Unsupported send mode.");
    static constexpr ParameterType parameter_type = ParameterType::send_mode; ///< The parameter type.
    using send_mode                               = SendModeTag;              ///< The send mode.
};

struct any_tag_t {}; ///< tag struct for message tag

/// @brief Possible types of tag
enum class TagType {
    value, ///< holds an actual value
    any    ///< special value MPI_ANY_TAG}
};

/// @brief Encapsulates a message tag.
/// @tparam The type of the tag.
template <TagType tag_type>
class TagParam {};

/// @brief Encapsulates a message tag. Specialization if an explicit tag value is provided.
template <>
class TagParam<TagType::value> : private ParameterObjectBase {
public:
    /// @param tag The tag.
    TagParam(int tag) : _tag_value(tag) {}
    static constexpr ParameterType parameter_type = ParameterType::tag; ///< The parameter type.
    static constexpr TagType       tag_type       = TagType::value;     ///< The tag type.
    /// @return The tag.
    [[nodiscard]] int tag() const {
        return _tag_value;
    }

private:
    int _tag_value; ///< the encapsulated tag value
};

/// @brief Encapsulates a message tag. Specialization if the value is MPI_ANY_TAG.
template <>
class TagParam<TagType::any> : private ParameterObjectBase {
public:
    static constexpr ParameterType parameter_type = ParameterType::tag; ///< The parameter type.
    static constexpr TagType       tag_type       = TagType::any;       ///< The tag type.
    /// @return The tag.
    [[nodiscard]] int tag() const {
        return MPI_ANY_TAG;
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

namespace tags {
static constexpr internal::any_tag_t any{}; ///< global constant for any tag
}

namespace rank {
static constexpr internal::rank_any_t  any{};  ///< global constant for any rank
static constexpr internal::rank_null_t null{}; ///< global constant for rank NULL
} // namespace rank

} // namespace kamping
