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

/// @brief Class representing a node in an (globally) aggregated tree, i.e., a node of a timer (or counter) tree
/// where the global aggregation operations has been performed and which can be printed.
///
/// @tparam DataType  Underlying data type.
template <typename DataType>
class AggregatedTreeNode : public internal::TreeNode<AggregatedTreeNode<DataType>> {
public:
    using internal::TreeNode<AggregatedTreeNode<DataType>>::TreeNode;

    ///@brief Type into which the aggregated data is stored together with the applied aggregation operation.
    using StorageType = std::unordered_map<GlobalAggregationMode, std::vector<ScalarOrContainer<DataType>>>;

    /// @brief Access to stored aggregated data.
    /// @return Reference to aggregated data.
    auto const& aggregated_data() const {
        return _aggregated_data;
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_mode Aggregation mode that has been applied to the data.
    /// @param data Scalar resulted from applying the given aggregation operation.
    void add(GlobalAggregationMode aggregation_mode, std::optional<DataType> data) {
        if (data) {
            _aggregated_data[aggregation_mode].emplace_back(data.value());
        }
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_mode Aggregation mode that has been applied to the duration data.
    /// @param data Vector of Scalars resulted from applying the given aggregation operation.
    void add(GlobalAggregationMode aggregation_mode, std::vector<DataType> const& data) {
        _aggregated_data[aggregation_mode].emplace_back(data);
    }

public:
    StorageType _aggregated_data; ///< Storage of the aggregated data.
};

/// @brief Class representing an aggregated measurement tree, i.e., a measurement tree for which the global aggregation
/// has been performed.
///
/// @tparam DataType Type of interanlly stored data.
template <typename DataType>
class AggregatedTree {
public:
    /// @brief Globally aggregates the measurement tree provided with \param measurement_root_node across all ranks in
    /// \param comm .
    ///
    /// @tparam MeasurementNode Type of the measurement tree to aggregate.
    /// @tparam Communicator Communicator defining the scope for the global aggregation.
    template <typename MeasurementNode, typename Communicator>
    AggregatedTree(MeasurementNode const& measurement_root_node, Communicator const& comm) : _root{"root"} {
        aggregate(_root, measurement_root_node, comm);
    }

    /// @brief Access to the root of the aggregated tree.
    /// @return Reference to root node of aggregated tree.
    auto& root() {
        return _root;
    }

    /// @brief Access to the root of the aggregated tree.
    /// @return Reference to root node of aggregated tree.
    auto const& root() const {
        return _root;
    }

private:
    AggregatedTreeNode<DataType> _root; ///< Root node of aggregated tree.
    /// @brief Traverses and evaluates the given (Measurement)TreeNode and stores the result in the corresponding
    /// AggregatedTreeNode
    ///
    /// param aggregation_tree_node Node where the aggregated data points are stored.
    /// param measurement_tree_node Node where the raw (not aggregated) data points are stored.
    template <typename MeasurementNode, typename Communciator>
    void aggregate(
        AggregatedTreeNode<DataType>& aggregation_tree_node,
        MeasurementNode&              measurement_tree_node,
        Communciator const&           comm
    ) {
        KASSERT(
            internal::is_string_same_on_all_ranks(measurement_tree_node.name(), comm),
            "Currently processed MeasurementTreeNode has not the same name on all ranks -> measurement trees have "
            "diverged",
            assert::heavy_communication
        );
        KASSERT(
            comm.is_same_on_all_ranks(measurement_tree_node.measurements().size()),
            "Currently processed MeasurementTreeNode has not the same number of measurements on all ranks -> "
            "measurement trees have "
            "diverged",
            assert::light_communication
        );

        // gather all durations at once as gathering all durations individually may deteriorate
        // the performance of the evaluation operation significantly.
        auto       recv_buf      = comm.gatherv(send_buf(measurement_tree_node.measurements()));
        auto const num_durations = measurement_tree_node.measurements().size();
        for (size_t duration_idx = 0; duration_idx < num_durations; ++duration_idx) {
            if (!comm.is_root()) {
                continue;
            }
            std::vector<DataType> cur_durations;
            cur_durations.reserve(comm.size());
            // gather the durations belonging to the same measurement
            for (size_t rank = 0; rank < comm.size(); ++rank) {
                cur_durations.push_back(recv_buf[duration_idx + rank * num_durations]);
            }

            for (auto const& aggregation_mode: measurement_tree_node.measurements_aggregation_operations()) {
                aggregate_measurements_globally(aggregation_mode, cur_durations, aggregation_tree_node);
            }
        }
        for (auto& measurement_tree_child: measurement_tree_node.children()) {
            auto& aggregation_tree_child = aggregation_tree_node.find_or_insert(measurement_tree_child->name());
            aggregate(aggregation_tree_child, *measurement_tree_child.get(), comm);
        }
    }

    /// @brief Computes the specified aggregation operation on an already gathered range of values.
    ///
    /// @param mode Aggregation operation to perform.
    /// @param gathered_data Durations gathered from all participating ranks.
    /// @param evaluation_node Object where the aggregated and evaluated measurements are stored.
    void aggregate_measurements_globally(
        GlobalAggregationMode                                mode,
        std::vector<DataType> const&                         gathered_data,
        kamping::measurements::AggregatedTreeNode<DataType>& evaluation_node
    ) {
        switch (mode) {
            case GlobalAggregationMode::max: {
                using Operation = internal::Max;
                evaluation_node.add(mode, Operation::compute(gathered_data));
                break;
            }
            case GlobalAggregationMode::min: {
                using Operation = internal::Min;
                evaluation_node.add(mode, Operation::compute(gathered_data));
                break;
            }
            case GlobalAggregationMode::sum: {
                using Operation = internal::Sum;
                evaluation_node.add(mode, Operation::compute(gathered_data));
                break;
            }
            case GlobalAggregationMode::gather: {
                using Operation = internal::Gather;
                evaluation_node.add(mode, Operation::compute(gathered_data));
                break;
            }
        }
    }
};

} // namespace kamping::measurements
