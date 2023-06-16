// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
/// @brief Defines utility classes for communicator creation using range based ranks descriptions.
///
#pragma once

#include <algorithm>
#include <cstddef>

namespace kamping {
/// @brief Struct encapsulating a MPI rank range triplet as used in functions like \c MPI_Group_range_incl/excl.
///
struct RankRange {
    int first;  ///< First rank contained in the rank range.
    int last;   ///< Last rank contained in the rank range.
    int stride; ///< Stride used in the rank range.
};

/// @brief RankRanges encapsulate multiple rank ranges which are used in functions like \c MPI_Group_range_incl/excl.
///
/// \c MPI_Group_range_incl/excl use multidimensional integer arrays to represent rank ranges. One can therefore use
/// plain c-style multi dimensional arrays directly which results in a zero-cost abstraction or use any container type
/// storing \c RankRange object which are then converted to the required plain c-style array.
class RankRanges {
public:
    /// @brief Constructor taking a plain two dimension c-style array.
    /// @param rank_range_array Pointer to int[3] representing contiguously stored plain MPI rank ranges.
    /// @param size Number of ranges stored in this array.
    RankRanges(int (*rank_range_array)[3], size_t size)
        : _is_lib_allocated{false},
          _rank_range_array{rank_range_array},
          _size{size} {}

    /// @brief Constructor taking any container storing \c RankRange objects.
    /// @param ranges Container storing \c RankRange objects.
    template <typename RangeContainer>
    RankRanges(RangeContainer const& ranges)
        : _is_lib_allocated{true},
          _rank_range_array{new int[ranges.size()][3]},
          _size{ranges.size()} {
        static_assert(
            std::is_same_v<typename RangeContainer::value_type, RankRange>,
            "Container's value_type must be RankRange!"
        );

        for (size_t i = 0; i < ranges.size(); ++i) {
            auto const& range       = ranges[i];
            _rank_range_array[i][0] = range.first;
            _rank_range_array[i][1] = range.last;
            _rank_range_array[i][2] = range.stride;
        }
    }

    ///@brief Destroys objects and deallocates any memory allocated during construction.
    ~RankRanges() {
        if (_is_lib_allocated) {
            delete[] _rank_range_array;
        }
    }

    /// @brief Get non-owning access to the underlying c-style array storing the rank ranges.
    /// @return Underlying c-style array of type int (*)[3].
    auto get() const {
        return _rank_range_array;
    }

    /// @brief Number of ranges stored in this object.
    /// @return Number of ranges.
    auto size() const {
        return _size;
    }

    /// @brief Checks whether the rank ranges contain a certain rank.
    /// @return Whether rank ranges contain a certain rank.
    bool contains(int rank) const {
        return std::any_of(_rank_range_array, _rank_range_array + _size, [&](auto const& plain_rank_range) {
            RankRange  rank_range{plain_rank_range[0], plain_rank_range[1], plain_rank_range[2]};
            bool const is_between_bounds     = rank >= rank_range.first && rank <= rank_range.last;
            int const  diff_to_start         = rank - rank_range.first;
            bool const is_multiple_of_stride = diff_to_start % rank_range.stride == 0;
            return is_between_bounds && is_multiple_of_stride;
        });
    }

private:
    bool const _is_lib_allocated; ///< Flag indicating whether the array needs to be freed upon object construction.
    int (*_rank_range_array)[3];  ///< Underlying c-style array of type int (*)[3].
    size_t _size;                 ///< Number of ranges stored in this object.
};
} // namespace kamping
