// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
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
/// This file contains the communication-free helpers used to aggregate a measurement tree with a number of collectives
/// which does not depend on the number of tree nodes.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kamping::measurements::internal {

/// @brief Default upper bound (in bytes) on the amount of measurement data gathered at the root rank per collective.
constexpr std::size_t default_max_bytes_at_root = 50 * 1000 * 1000;

/// @brief Describes where the datapoints of a single measurement tree node live in the flattened datapoint array and
/// which node of the aggregated tree they belong to.
///
/// @tparam MeasurementNode Type of the node of the measurement tree.
/// @tparam AggregatedNode Type of the node of the aggregated tree.
template <typename MeasurementNode, typename AggregatedNode>
struct MeasurementSlot {
    MeasurementNode const* measurement_node; ///< Node of the measurement tree the datapoints stem from.
    AggregatedNode*        aggregated_node;  ///< Node of the aggregated tree the results are stored in.
    std::size_t            offset;           ///< Index of the node's first datapoint in the flattened array.
    std::size_t            dim;              ///< Number of datapoints stored at the node.
};

/// @brief Incremental FNV-1a hash.
///
/// A hash with an explicitly specified algorithm is used instead of \c std::hash as the hash values are compared
/// across ranks.
class Fnv1aHasher {
public:
    /// @brief Adds the given 64 bit value to the hash.
    /// @param value Value to add to the hash.
    void update(std::uint64_t value) {
        for (std::size_t i = 0; i < sizeof(value); ++i) {
            update_byte(static_cast<unsigned char>((value >> (8 * i)) & 0xffu));
        }
    }

    /// @brief Adds the given string to the hash.
    /// @param value String to add to the hash.
    void update(std::string const& value) {
        update(static_cast<std::uint64_t>(value.size()));
        for (char const c: value) {
            update_byte(static_cast<unsigned char>(c));
        }
    }

    /// @brief Access to the current hash value.
    /// @return The hash of everything added so far.
    std::uint64_t digest() const {
        return _state;
    }

private:
    /// @brief Adds a single byte to the hash.
    /// @param byte Byte to add to the hash.
    void update_byte(unsigned char byte) {
        _state ^= static_cast<std::uint64_t>(byte);
        _state *= 0x100000001b3ull;
    }

    std::uint64_t _state{0xcbf29ce484222325ull}; ///< Current hash value.
};

/// @brief Result of flattening a measurement tree, see flatten_measurement_tree().
///
/// @tparam DataType Type of the datapoints stored in the measurement tree.
/// @tparam MeasurementNode Type of the node of the measurement tree.
/// @tparam AggregatedNode Type of the node of the aggregated tree.
template <typename DataType, typename MeasurementNode, typename AggregatedNode>
struct FlattenedMeasurementTree {
    using Slot = MeasurementSlot<MeasurementNode, AggregatedNode>; ///< Type describing a single node's datapoints.

    std::vector<DataType> values;   ///< Datapoints of all nodes concatenated in depth-first order.
    std::vector<Slot>     layout;   ///< One entry per node of the measurement tree, in depth-first order.
    std::uint64_t structure_hash{}; ///< Hash over the node count, the datapoint count and all (name, dim) pairs.
};

/// @brief Computes the hash describing the structure of a single measurement tree node.
///
/// @param name Name of the node.
/// @param dim Number of datapoints stored at the node.
/// @return Hash over the given name and dimension.
inline std::uint64_t hash_node(std::string const& name, std::size_t dim) {
    Fnv1aHasher hasher;
    hasher.update(name);
    hasher.update(static_cast<std::uint64_t>(dim));
    return hasher.digest();
}

/// @brief Appends the given measurement tree node and its descendants to the given flattened measurement tree.
///
/// @tparam DataType Type of the datapoints stored in the measurement tree.
/// @tparam MeasurementNode Type of the node of the measurement tree.
/// @tparam AggregatedNode Type of the node of the aggregated tree.
/// @param measurement_node Node of the measurement tree to append.
/// @param aggregated_node Node of the aggregated tree corresponding to \p measurement_node.
/// @param flattened Flattened measurement tree to append to.
/// @param hasher Hasher to add the traversed nodes' structure to.
template <typename DataType, typename MeasurementNode, typename AggregatedNode>
void flatten_subtree(
    MeasurementNode const&                                               measurement_node,
    AggregatedNode&                                                      aggregated_node,
    FlattenedMeasurementTree<DataType, MeasurementNode, AggregatedNode>& flattened,
    Fnv1aHasher&                                                         hasher
) {
    auto const& measurements = measurement_node.measurements();
    flattened.layout.push_back({&measurement_node, &aggregated_node, flattened.values.size(), measurements.size()});
    flattened.values.insert(flattened.values.end(), measurements.begin(), measurements.end());
    hasher.update(hash_node(measurement_node.name(), measurements.size()));

    for (auto const& measurement_child: measurement_node.children()) {
        auto& aggregated_child = aggregated_node.find_or_insert(measurement_child->name());
        flatten_subtree<DataType>(*measurement_child, aggregated_child, flattened, hasher);
    }
}

/// @brief Traverses the given measurement tree depth-first, concatenates the datapoints of all nodes and creates the
/// corresponding node in the aggregated tree.
///
/// A node is mirrored into the aggregated tree even if it does not store any datapoints, as the aggregated tree
/// determines the structure of the printed output.
///
/// @tparam DataType Type of the datapoints stored in the measurement tree.
/// @tparam MeasurementNode Type of the node of the measurement tree.
/// @tparam AggregatedNode Type of the node of the aggregated tree.
/// @param measurement_root_node Root of the measurement tree to flatten.
/// @param aggregated_root_node Root of the aggregated tree to mirror the measurement tree into.
/// @return The concatenated datapoints, the layout describing which datapoints belong to which node and a hash over
/// the structure of the measurement tree.
template <typename DataType, typename MeasurementNode, typename AggregatedNode>
FlattenedMeasurementTree<DataType, MeasurementNode, AggregatedNode>
flatten_measurement_tree(MeasurementNode const& measurement_root_node, AggregatedNode& aggregated_root_node) {
    FlattenedMeasurementTree<DataType, MeasurementNode, AggregatedNode> flattened;
    Fnv1aHasher                                                         hasher;

    flatten_subtree<DataType>(measurement_root_node, aggregated_root_node, flattened, hasher);

    hasher.update(static_cast<std::uint64_t>(flattened.layout.size()));
    hasher.update(static_cast<std::uint64_t>(flattened.values.size()));
    flattened.structure_hash = hasher.digest();

    return flattened;
}

/// @brief Computes the number of datapoints per rank which are gathered at the root rank with a single collective.
///
/// At least one datapoint per rank is gathered, i.e. \p max_bytes_at_root is exceeded if a single datapoint per rank
/// already does not fit into the given budget.
///
/// @param comm_size Number of ranks contributing datapoints.
/// @param bytes_per_datapoint Size of a single datapoint in bytes.
/// @param max_bytes_at_root Upper bound on the number of bytes gathered at the root rank per collective.
/// @return Number of datapoints each rank contributes to a single gather.
inline std::size_t
compute_batch_size(std::size_t comm_size, std::size_t bytes_per_datapoint, std::size_t max_bytes_at_root) {
    return std::max<std::size_t>(1, max_bytes_at_root / (comm_size * bytes_per_datapoint));
}

} // namespace kamping::measurements::internal
