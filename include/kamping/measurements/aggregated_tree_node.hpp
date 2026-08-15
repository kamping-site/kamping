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

#include <array>
#include <cstddef>
#include <cstdint>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/kassert/kassert.hpp"
#include "kamping/measurements/internal/flatten.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"
#include "kamping/measurements/measurement_aggregation_definitions.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/span.hpp"

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
    /// @brief Tag to select the node-by-node aggregation implementation, see the corresponding constructor.
    struct UnbatchedTag {};

    /// @brief Globally aggregates the measurement tree provided with \param measurement_root_node across all ranks in
    /// \param comm .
    ///
    /// The datapoints of all nodes are gathered at the root rank in batches of at most \p max_bytes_at_root bytes,
    /// i.e. the number of collective operations depends on the amount of data received at the root rank and not on
    /// the number of nodes in the measurement tree.
    ///
    /// The measurement tree must have the same structure and the same number of datapoints per node on all ranks.
    /// This is verified using a single collective operation; if the trees have diverged, all ranks throw (or fail an
    /// assertion if exception mode is disabled). As for all other error checking in KaMPIng, this verification is
    /// compiled out when building with assertion level \c none.
    ///
    /// @tparam MeasurementNode Type of the measurement tree to aggregate.
    /// @tparam CommunicatorType Communicator defining the scope for the global aggregation.
    /// @param measurement_root_node Root of the measurement tree to aggregate.
    /// @param comm Communicator defining the scope for the global aggregation.
    /// @param max_bytes_at_root Upper bound on the number of bytes gathered at the root rank per collective
    /// operation. Must be the same on all ranks as it determines the number of collective operations issued.
    template <typename MeasurementNode, typename CommunicatorType>
    AggregatedTree(
        MeasurementNode const&  measurement_root_node,
        CommunicatorType const& comm,
        std::size_t             max_bytes_at_root = internal::default_max_bytes_at_root
    )
        : _root{"root"} {
        aggregate_batched(measurement_root_node, comm, max_bytes_at_root);
    }

    /// @brief Globally aggregates the measurement tree node by node, using two collective operations per node.
    ///
    /// This is the predecessor of the batched aggregation and is retained to validate the latter against it.
    ///
    /// @tparam MeasurementNode Type of the measurement tree to aggregate.
    /// @tparam CommunicatorType Communicator defining the scope for the global aggregation.
    /// @param measurement_root_node Root of the measurement tree to aggregate.
    /// @param comm Communicator defining the scope for the global aggregation.
    template <typename MeasurementNode, typename CommunicatorType>
    AggregatedTree(UnbatchedTag, MeasurementNode const& measurement_root_node, CommunicatorType const& comm)
        : _root{"root"} {
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

    /// @brief Flattens the given measurement tree, verifies that it is identical on all ranks and aggregates its
    /// datapoints using gather operations of bounded size.
    ///
    /// @tparam MeasurementNode Type of the measurement tree to aggregate.
    /// @tparam CommunicatorType Communicator defining the scope for the global aggregation.
    /// @param measurement_root_node Root of the measurement tree to aggregate.
    /// @param comm Communicator defining the scope for the global aggregation.
    /// @param max_bytes_at_root Upper bound on the number of bytes gathered at the root rank per collective
    /// operation.
    template <typename MeasurementNode, typename CommunicatorType>
    void aggregate_batched(
        MeasurementNode const& measurement_root_node, CommunicatorType const& comm, std::size_t max_bytes_at_root
    ) {
        auto const flattened = internal::flatten_measurement_tree<DataType>(measurement_root_node, _root);

        internal::Fnv1aHasher hasher;
        hasher.update(flattened.structure_hash);
        hasher.update(static_cast<std::uint64_t>(max_bytes_at_root));
        assert_measurement_trees_are_identical(hasher.digest(), flattened, comm);

        std::size_t const num_datapoints = flattened.values.size();
        std::size_t const batch_size = internal::compute_batch_size(comm.size(), sizeof(DataType), max_bytes_at_root);

        std::vector<DataType> gather_buffer;
        std::vector<DataType> gathered_data;
        if (comm.is_root()) {
            gather_buffer.resize(std::min(batch_size, num_datapoints) * comm.size());
            gathered_data.reserve(comm.size());
        }

        std::size_t slot_idx = 0;
        for (std::size_t offset = 0; offset < num_datapoints; offset += batch_size) {
            std::size_t const cur_batch_size = std::min(batch_size, num_datapoints - offset);
            comm.gather(
                send_buf(Span<DataType const>(flattened.values.data() + offset, cur_batch_size)),
                send_count(asserting_cast<int>(cur_batch_size)),
                recv_count(asserting_cast<int>(cur_batch_size)),
                recv_buf(gather_buffer)
            );
            if (!comm.is_root()) {
                continue;
            }
            for (std::size_t datapoint_idx = 0; datapoint_idx < cur_batch_size; ++datapoint_idx) {
                gathered_data.clear();
                for (std::size_t rank = 0; rank < comm.size(); ++rank) {
                    gathered_data.push_back(gather_buffer[rank * cur_batch_size + datapoint_idx]);
                }
                while (offset + datapoint_idx >= flattened.layout[slot_idx].offset + flattened.layout[slot_idx].dim) {
                    ++slot_idx;
                }
                auto const& slot = flattened.layout[slot_idx];
                for (auto const& aggregation_mode: slot.measurement_node->measurements_aggregation_operations()) {
                    aggregate_measurements_globally(aggregation_mode, gathered_data, *slot.aggregated_node);
                }
            }
        }
    }

    /// @brief Verifies that the given hash describing the flattened measurement tree is the same on all ranks.
    ///
    /// @tparam FlattenedTree Type of the flattened measurement tree.
    /// @tparam CommunicatorType Communicator defining the scope for the global aggregation.
    /// @param hash Hash describing this rank's measurement tree.
    /// @param flattened Flattened measurement tree, used to report where the trees have diverged.
    /// @param comm Communicator defining the scope for the global aggregation.
    template <typename FlattenedTree, typename CommunicatorType>
    void assert_measurement_trees_are_identical(
        std::uint64_t hash, FlattenedTree const& flattened, CommunicatorType const& comm
    ) {
        std::array<std::uint64_t, 2> const local_hashes{hash, ~hash};
        auto const global_hashes = comm.allreduce(send_buf(local_hashes), op(ops::max<std::uint64_t>{}));
        if (global_hashes[0] == hash && ~global_hashes[1] == hash) {
            return;
        }
        report_diverged_measurement_trees(flattened, comm);
    }

    /// @brief Determines where the measurement trees have diverged and reports it.
    ///
    /// This is only called once a divergence has been detected on all ranks, i.e. it is not part of the fast path.
    ///
    /// @tparam FlattenedTree Type of the flattened measurement tree.
    /// @tparam CommunicatorType Communicator defining the scope for the global aggregation.
    /// @param flattened Flattened measurement tree of this rank.
    /// @param comm Communicator defining the scope for the global aggregation.
    template <typename FlattenedTree, typename CommunicatorType>
    void report_diverged_measurement_trees(FlattenedTree const& flattened, CommunicatorType const& comm) {
        std::size_t const num_nodes = flattened.layout.size();
        THROWING_KAMPING_ASSERT(
            comm.is_same_on_all_ranks(num_nodes),
            "Measurement trees have diverged: this rank's tree has " << num_nodes << " nodes, other ranks disagree."
        );

        std::vector<std::uint64_t> node_hashes;
        node_hashes.reserve(num_nodes);
        for (auto const& slot: flattened.layout) {
            node_hashes.push_back(internal::hash_node(slot.measurement_node->name(), slot.dim));
        }
        auto const max_hashes = comm.allreduce(send_buf(node_hashes), op(ops::max<std::uint64_t>{}));
        auto const min_hashes = comm.allreduce(send_buf(node_hashes), op(ops::min<std::uint64_t>{}));
        for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
            if (max_hashes[node_idx] != min_hashes[node_idx]) {
                auto const& slot = flattened.layout[node_idx];
                THROWING_KAMPING_ASSERT(
                    false,
                    "Measurement trees have diverged at node " << node_idx << ", which is named \""
                                                               << slot.measurement_node->name() << "\" and holds "
                                                               << slot.dim << " datapoint(s) on this rank."
                );
            }
        }
        THROWING_KAMPING_ASSERT(false, "Measurement trees have diverged.");
    }

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
        KAMPING_ASSERT(
            internal::is_string_same_on_all_ranks(measurement_tree_node.name(), comm),
            "Currently processed MeasurementTreeNode has not the same name on all ranks -> measurement trees have "
            "diverged",
            assert::heavy_communication
        );
        KAMPING_ASSERT(
            comm.is_same_on_all_ranks(measurement_tree_node.measurements().size()),
            "Currently processed MeasurementTreeNode has not the same number of measurements on all ranks -> "
            "measurement trees have "
            "diverged",
            assert::light_communication
        );

        // gather all durations at once as gathering all durations individually may deteriorate
        // the performance of the evaluation operation significantly.
        auto       recv_buf      = comm.gather(send_buf(measurement_tree_node.measurements()));
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
