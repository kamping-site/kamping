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

#include <mpi.h>
#include <kassert/kassert.hpp>

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
  using DataType = std::int64_t;
  /// @brief Constructs a timer using the \c MPI_COMM_WORLD communicator.
  Counter() : _tree{}, _comm{comm_world()} {}

  /// @brief Constructs a timer using a given communicator.
  ///
  /// @param comm Communicator in which the time measurements are executed.
  Counter(CommunicatorType const& comm) : _tree{}, _comm{comm} {}

  /// @brief Stops the currently active measurement and store the result.
  /// @param duration_aggregation_modi Specify how the measurement duration is
  /// aggregated over all participationg PEs when Timer::aggregate() is called.
  void add(std::string const& name, DataType const& data,
           std::vector<DataAggregationMode> const& data_aggregation_modi =
               std::vector<DataAggregationMode>{}) {
    add_impl(name, data, KeyAggregationMode::accumulate, data_aggregation_modi);
  }

  /// @brief Stops the currently active measurement and store the result. If the
  /// key associated with the measurement that is stopped has already been used
  /// at the current hierarchy level the duration is append to a list of
  /// previously measured durations at this hierarchy level with the same key.
  /// @param duration_aggregation_modi Specify how the measurement duration is
  /// aggregated over all participationg PEs when Timer::aggregate() is called.
  void append(std::string const& name, DataType const& data,
              std::vector<DataAggregationMode> const& add_aggregation_modi =
                  std::vector<DataAggregationMode>{}) {
    add_impl(name, data, KeyAggregationMode::append, add_aggregation_modi);
  }

  /// @brief Evaluates the time measurements represented by the timer tree for
  /// which the current node is the root node globally. The measured durations
  /// are aggregated over all participating PEs and the result is stored at the
  /// root rank of the given communicator. The used aggregation operations can
  /// be specified via TimerTreeNode::data_aggregation_operations(). The
  /// durations are aggregated node by node.
  ///
  /// @return Root of the evaluation tree which encapsulated the aggregated data
  /// in a tree structure representing the measurements.
  auto evaluate() {
    EvaluationTreeNode<DataType> root("counter");
    evaluate(root, _tree.root);
    return root;
  }

  /// @brief Clears all stored measurements.
  void clear() { _tree.reset(); }

  /// @brief Aggregates and outputs the the executed measurements. The output is
  /// done via the print() method of a given Printer object.
  ///
  /// The print() method must accept an object of type EvaluationTreeNode and
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
  void evaluate_and_print(Printer&& printer) {
    auto evaluation_tree_root = evaluate();
    if (_comm.is_root()) {
      printer.print(evaluation_tree_root);
    }
  }

 private:
  internal::Tree<internal::CounterTreeNode<DataType>> _tree;
  const CommunicatorType&
      _comm;  ///< Communicator in which the time measurements take place.

  /// @brief Stops the currently active measurement and store the result.
  /// @param key_aggregation_mode Specifies how the measurement duration is
  /// locally aggregated when there are multiple measurements at the same level
  /// with identical key.
  /// @param data_aggregation_modi Specifies how the measurement data is
  /// aggregated over all participationg ranks when Timer::aggregate() is
  /// called.
  void add_impl(
      std::string const& name, DataType const& data,
      LocalAggregationMode key_aggregation_mode,
      std::vector<GlobalAggregationMode> const& data_aggreation_modi) {
    auto child = _tree.current_node->find_or_insert(name);
    child.aggregate_measurements_locally(data, key_aggregation_mode);
    if (!data_aggreation_modi.empty()) {
      child.measurements_aggregation_operations() = data_aggreation_modi;
    }
  }

  /// @brief Traverses and evaluates the given TimerTreeNode and stores the
  /// result in the corresponding EvaluationTreeNode
  ///
  /// param evaluation_tree_node Node where the aggregated durations are stored.
  /// param timer_tree_node Node where the raw durations are stored.
  void evaluate(
      internal::EvaluationTreeNode<DataType>& evaluation_tree_node,
      internal::CounterTreeNode<DataType>& counter_tree_node) {
    KASSERT(is_string_same_on_all_ranks(counter_tree_node.name(), _comm),
            "Currently processed TimerTreeNode has not the same name on all "
            "ranks -> timers have diverged",
            assert::heavy_communication);
    KASSERT(
        _comm.is_same_on_all_ranks(counter_tree_node.measurements().size()),
        "Currently processed TimerTreeNode has not the same number of "
        "measurements on all ranks -> timers have "
        "diverged",
        assert::light_communication);

    auto recv_buf = _comm.gatherv(send_buf(counter_tree_node.measurements()));

    const auto num_data = counter_tree_node.measurements().size();
    for (size_t i = 0; i < counter_tree_node.measurements().size(); ++i) {
      if (!_comm.is_root()) {
        continue;
      }
      std::vector<DataType> cur_gathered_measurements;
      cur_gathered_measurements.reserve(_comm.size());
      for (std::size_t j = 0; j < _comm.size(); ++j) {
        cur_gathered_measurements.push_back(recv_buf[i + j * num_data]);
      }

      for (auto const& aggregation_mode :
           counter_tree_node.measurements_aggregation_operations()) {
        aggregate_measurements_globally(
            aggregation_mode, cur_gathered_measurements, evaluation_tree_node);
      }
    }
    for (auto& measurement_tree_child : counter_tree_node.children()) {
      auto evaluation_tree_child =
          evaluation_tree_node.find_or_insert(measurement_tree_child->name());
      evaluate(evaluation_tree_child, *measurement_tree_child.get());
    }
  }

  /// @brief Computes the specified aggregation operation on an already gathered
  /// range of durations.
  ///
  /// @param mode Aggregation operation to perform.
  /// @param gathered_data Durations gathered from all participating ranks.
  /// @param evaluation_node Object where the aggregated and evaluated durations
  /// are stored.
  void aggregate_measurements_globally(
      GlobalAggregationMode mode, std::vector<DataType> const& gathered_data,
      kamping::measurements::AggregatedTreeNode<DataType>& evaluation_node) {
    switch (mode) {
      case GlobalAggregationMode::max: {
        using Operation = internal::Max;
        evaluation_node.add(mode,
                            Operation::compute(gathered_data));
        break;
      }
      case GlobalAggregationMode::min: {
        using Operation = internal::Min;
        evaluation_node.add(Operation::operation_name(),
                            Operation::compute(gathered_data));
        break;
      }
      case GlobalAggregationMode::sum: {
        using Operation = internal::Sum;
        evaluation_node.add(Operation::operation_name(),
                            Operation::compute(gathered_data));
        break;
      }
      case GlobalAggregationMode::gather: {
        using Operation = internal::Gather;
        evaluation_node.add(Operation::operation_name(),
                            Operation::compute(gathered_data));
        break;
      }
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
}  // namespace kamping::measurements
