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
/// This file contains a tree node class which can be used to represent an evaluated measurement tree.

#pragma once
#include "kamping/measurements/internal/measurement_utils.hpp"
#include "kamping/measurements/measurement_aggregation_definitions.hpp"

namespace kamping::measurements {
/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the
///  same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam Duration  Type of a duration.
template <typename Duration>
class AggregatedTreeNode : public internal::TreeNode<AggregatedTreeNode<Duration>> {
public:
    using internal::TreeNode<AggregatedTreeNode<Duration>>::TreeNode;

    ///@brief Type into which the aggregated data is stored together with the applied aggregation operation.
    using StorageType = std::unordered_map<GlobalAggregationMode, std::vector<ScalarOrContainer<Duration>>>;

    /// @brief Access to stored aggregated data.
    /// @return Reference to aggregated data.
    auto const& aggregated_data() const {
        return _aggregated_data;
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_mode Aggregation mode that has been applied to the duration data.
    /// @param data Scalar resulted from applying the given aggregation operation.
    void add(GlobalAggregationMode aggregation_mode, std::optional<Duration> data) {
        if (data) {
            _aggregated_data[aggregation_mode].emplace_back(data.value());
        }
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_mode Aggregation mode that has been applied to the duration data.
    /// @param data Vector of Scalars resulted from applying the given aggregation operation.
    void add(GlobalAggregationMode aggregation_mode, std::vector<Duration> const& data) {
        _aggregated_data[aggregation_mode].emplace_back(data);
    }

public:
    StorageType _aggregated_data; ///< Storage of the aggregated data.
};
} // namespace kamping::measurements
