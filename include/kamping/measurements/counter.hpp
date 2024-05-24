// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
/// This file contains a (distributed) counter class.

#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/measurements/aggregated_tree_node.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"

namespace kamping::measurements {

/// @brief Distributed counter object.
/// @tparam CommunicatorType Communicator in which the measurements are
/// executed.
template <typename CommunicatorType = Communicator<>>
class Counter {
public:
    using DataType = std::int64_t; ///< Data type of the stored measurements.
    /// @brief Constructs a timer using the \c MPI_COMM_WORLD communicator.
    Counter() : _tree{}, _comm{comm_world()} {}

    /// @brief Constructs a timer using a given communicator.
    ///
    /// @param comm Communicator in which the time measurements are executed.
    Counter(CommunicatorType const& comm) : _tree{}, _comm{comm} {}

    /// @brief Creates a measurement entry with name \param name and stores \param data therein. If such an entry
    /// already exists with associated data entry `data_prev`, \c data will be added to it, i.e. `data_prev +
    /// data`.
    /// @param global_aggregation_modi Specify how the measurement entry is aggregated over all participationg PEs when
    /// Counter::aggregate() is called.
    void
    add(std::string const&                        name,
        DataType const&                           data,
        std::vector<GlobalAggregationMode> const& global_aggregation_modi = std::vector<GlobalAggregationMode>{}) {
        add_measurement(name, data, LocalAggregationMode::accumulate, global_aggregation_modi);
    }

    /// @brief Looks for a measurement entry with name \param name and appends \param data to the list of previously
    /// stored data. If no such entry exists, a new measurement entry with \c data as first entry will be created. entry
    /// `data_prev`, \c data will be added to it, i.e. `data_prev + data`.
    /// @param global_aggregation_modi Specify how the measurement entry is aggregated over all participationg PEs when
    /// Counter::aggregate() is called.
    void append(
        std::string const&                        name,
        DataType const&                           data,
        std::vector<GlobalAggregationMode> const& global_aggregation_modi = std::vector<GlobalAggregationMode>{}
    ) {
        add_measurement(name, data, LocalAggregationMode::append, global_aggregation_modi);
    }

    /// @brief Aggregate the measurement entries globally.
    /// @return AggregatedTree object which encapsulates the aggregated data in a tree structure representing the
    /// measurements.
    auto aggregate() {
        AggregatedTree<DataType> aggregated_tree(_tree.root, _comm);
        return aggregated_tree;
    }

    /// @brief Clears all stored measurements.
    void clear() {
        _tree.reset();
    }

    /// @brief Aggregates and outputs the the executed measurements. The output is
    /// done via the print() method of a given Printer object.
    ///
    /// The print() method must accept an object of type AggregatedTreeNode and
    /// receives the root of the evaluated timer tree as parameter. The print()
    /// method is only called on the root rank of the communicator. See
    /// EvaluationTreeNode for the accessible data. The
    /// EvaluationTreeNode::children() member function can be used to navigate the
    /// nested measurement structure.
    ///
    /// @tparam Printer Type of printer which is used to output the aggregated
    /// timing data. Printer must possess a member print() which accepts a
    /// EvaluationTreeNode as parameter.
    /// @param printer Printer object used to output the aggregated timing data.
    template <typename Printer>
    void aggregate_and_print(Printer&& printer) {
        auto const aggregated_tree = aggregate();
        if (_comm.is_root()) {
            printer.print(aggregated_tree.root());
        }
    }

private:
    internal::Tree<internal::CounterTreeNode<DataType>>
        _tree; ///< Tree structure in which the counted values are stored. Note that unlike for Timer, the tree is
               ///< always a star as there is currently no functionality to allow for "nested" counting, e.g. by
               ///< defining different phase within your algorithm.
    CommunicatorType const& _comm; ///< Communicator in which the time measurements take place.

    /// @brief Adds a new measurement to the tree
    /// @param local_aggregation_mode Specifies how the measurement duration is
    /// locally aggregated when there are multiple measurements at the same level
    /// with identical key.
    /// @param global_aggregation_modi Specifies how the measurement data is
    /// aggregated over all participationg ranks when Timer::aggregate() is
    /// called.
    void add_measurement(
        std::string const&                        name,
        DataType const&                           data,
        LocalAggregationMode                      local_aggregation_mode,
        std::vector<GlobalAggregationMode> const& global_aggreation_modi
    ) {
        auto& child = _tree.current_node->find_or_insert(name);
        child.aggregate_measurements_locally(data, local_aggregation_mode);
        if (!global_aggreation_modi.empty()) {
            child.measurements_aggregation_operations() = global_aggreation_modi;
        }
    }
};

/// @brief A basic Counter that uses kamping::Communicator<> as underlying
/// communicator type.
using BasicCounter = Counter<Communicator<>>;

/// @brief Gets a reference to a kamping::measurements::BasicTimer.
///
/// @return A reference to a kamping::measurements::BasicCounter.
inline Counter<Communicator<>>& counter() {
    static Counter<Communicator<>> counter;
    return counter;
}
} // namespace kamping::measurements
