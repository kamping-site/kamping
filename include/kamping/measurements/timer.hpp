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
/// This file contains a (distributed) timer class.

#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/measurements/measurement_utils.hpp"

namespace kamping::measurements {

/// @brief Distributed timer object.
///
/// The timer is able to execute time measurements in a hierarchical manner.
/// Time measurements are executed with a matching pair of calls to Timer::start() and Timer::stop().
/// Each call to start() enters a new level in the hierarchy which is left again with the corresponding call to
/// stop(). Time measurements can be nested and the parent measurement remains active while its child time
/// measurment take place.
///
/// See the following pseudocode example:
///
/// \code
/// Timer timer;
/// timer.start("algorithm");
///   timer.start("preprocessing");
///   timer.stop();                       // stops "preprocessing" measurement
///     timer.start("core_algorithm");
///       timer.start("subroutine");
///       timer.stop();                   // stops "subroutine" measurement
///   timer.stop();                       // stops "core_algorithm" measurement
///   timer.start("postprocessing");
///   timer.stop();                       // stops "postprocessing" measurement
/// timer.stop();                         // stops "algorithm" measurement
/// \endcode
///
/// This corresponds to the following timing hierarchy:
///
/// \code
/// // Measurement key          Duration
/// // ----------------------------------
/// // algorithm:...............6.0 sec
/// // |-- preprocessing:.......1.0 sec
/// // |-- core_algorithm:......4.0 sec
/// // |   `-- subroutine:......2.0 sec
/// // `-- postprocessing:......2.0 sec
/// \endcode
///
/// The order and argument of calls of start() and stop() must be identical on all ranks in the given communicator.
///
/// @tparam CommunicatorType Communicator in which the time measurements are executed.
template <typename CommunicatorType = Communicator<>>
class Timer {
public:
    /// @brief Constructs a timer using the \c MPI_COMM_WORLD communicator.
    Timer() : _timer_tree{}, _comm{comm_world()} {}

    /// @brief Constructs a timer using a given communicator.
    ///
    /// @param comm Communicator in which the time measurements are executed.
    Timer(CommunicatorType const& comm) : _timer_tree{}, _comm{comm} {}

    /// @brief Synchronizes all ranks in the underlying communicator via a barrier and then start the measuremt with
    /// the given key.
    /// @param key Key with which the started time measurement is associated.
    void synchronize_and_start(std::string const& key) {
        bool const use_barrier = true;
        start_impl(key, use_barrier);
    }

    /// @brief Starts the measuremt with the given key.
    /// @param key Key with which the started time measurement is associated.
    void start(std::string const& key) {
        bool const use_barrier = false;
        start_impl(key, use_barrier);
    }

    /// @brief Stops the currently active measurement and store the result.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop(std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}) {
        stop_impl(KeyAggregationMode::accumulate, duration_aggregation_modi);
    }

    /// @brief Stops the currently active measurement and store the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is added to
    /// the last measured duration at this hierarchy level with the same key.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop_and_accumulate(
        std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}
    ) {
        stop_impl(KeyAggregationMode::accumulate, duration_aggregation_modi);
    }

    /// @brief Stops the currently active measurement and store the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is append to a
    /// list of previously measured durations at this hierarchy level with the same key.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop_and_append(
        std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}
    ) {
        stop_impl(KeyAggregationMode::append, duration_aggregation_modi);
    }

    /// @brief Evaluates the time measurements represented by the timer tree for which the current node is the root
    /// node globally. The measured durations are aggregated over all participating PEs and the result is stored at
    /// the root rank of the given communicator. The used aggregation operations can be specified via
    /// TimerTreeNode::data_aggregation_operations().
    /// The durations are aggregated node by node.
    ///
    /// @return Root of the evaluation tree which encapsulated the aggregated data in a tree structure representing the
    /// measurements.
    auto evaluate() {
        EvaluationTreeNode<double> root("root");
        evaluate(root, _timer_tree.root);
        return root;
    }

    /// @brief Clears all stored measurements.
    void clear() {
        std::cout << "add root: " << &(_timer_tree.root) << std::endl;
        std::cout << "current : " << _timer_tree.current_node << std::endl;
        _timer_tree.reset();
        std::cout << "add root: " << &(_timer_tree.root) << std::endl;
        std::cout << "current : " << _timer_tree.current_node << std::endl;
    }

    /// @brief Aggregates and outputs the the executed measurements. The output is done via the print()
    /// method of a given Printer object.
    ///
    /// The print() method must accept an object of type EvaluationTreeNode and receives the root of the evaluated timer
    /// tree as parameter. The print() method is only called on the root rank of the communicator. See
    /// EvaluationTreeNode for the accessible data. The EvaluationTreeNode::children() member function can be used to
    /// navigate the nested measurement structure.
    ///
    /// @tparam Printer Type of printer which is used to output the aggregated timing data. Printer must possess a
    /// member print() which accepts a EvaluationTreeNode as parameter.
    /// @param printer Printer object used to output the aggregated timing data.
    template <typename Printer>
    void evaluate_and_print(Printer&& printer) {
        auto evaluation_tree_root = evaluate();
        if (comm_world().is_root()) {
            printer.print(evaluation_tree_root);
        }
    }

private:
    internal::TimerTree<double, double>
                            _timer_tree; ///< Timer tree used to represent the hiearchical time measurements.
    CommunicatorType const& _comm;       ///< Communicator in which the time measurements take place.

    /// @brief Starts a time measurement.
    void start_impl(std::string const& key, bool use_barrier) {
        auto node = _timer_tree.current_node->find_or_insert(key);
        if (use_barrier) {
            comm_world().barrier();
        }
        node->startpoint()       = Environment<>::wtime();
        _timer_tree.current_node = node;
    }

    /// @brief Stops the currently active measurement and store the result.
    /// @param key_aggregation_mode Specifies how the measurement duration is locally aggregated when there are multiple
    /// measurements at the same level with identical key.
    /// @param duration_aggregation_modi Specifies how the measurement duration is aggregated over all participationg
    /// ranks when Timer::aggregate() is called.
    void stop_impl(
        KeyAggregationMode key_aggregation_mode, std::vector<DataAggregationMode> const& duration_aggregation_modi
    ) {
        auto endpoint   = Environment<>::wtime();
        auto startpoint = _timer_tree.current_node->startpoint();
        _timer_tree.current_node->aggregate_measurements_locally(endpoint - startpoint, key_aggregation_mode);
        if (!duration_aggregation_modi.empty()) {
            _timer_tree.current_node->duration_aggregation_operations() = duration_aggregation_modi;
        }
        _timer_tree.current_node = _timer_tree.current_node->parent_ptr();
    }

    /// @brief Traverses and evaluates the given TimerTreeNode and stores the result in the corresponding
    /// EvaluationTreeNode
    ///
    /// param evaluation_tree_node Node where the aggregated durations are stored.
    /// param timer_tree_node Node where the raw durations are stored.
    void evaluate(
        EvaluationTreeNode<double>& evaluation_tree_node, internal::TimerTreeNode<double, double>& timer_tree_node
    ) {
        KASSERT(
            is_string_same_on_all_ranks(timer_tree_node.name(), _comm),
            "Currently processed TimerTreeNode has not the same name on all ranks -> timers have diverged",
            assert::heavy_communication
        );
        KASSERT(
            _comm.is_same_on_all_ranks(timer_tree_node.durations().size()),
            "Currently processed TimerTreeNode has not the same number of measurements on all ranks -> timers have "
            "diverged",
            assert::light_communication
        );
        for (auto const& item: timer_tree_node.durations()) {
            auto recv_buf = _comm.gather(send_buf(item)).extract_recv_buffer();
            if (!_comm.is_root()) {
                continue;
            }

            for (auto const& aggregation_mode: timer_tree_node.duration_aggregation_operations()) {
                aggregate_measurements_globally(aggregation_mode, recv_buf, evaluation_tree_node);
            }
        }
        for (auto& timer_tree_child: timer_tree_node.children()) {
            auto evaluation_tree_child = evaluation_tree_node.find_or_insert(timer_tree_child->name());
            evaluate(*evaluation_tree_child, *timer_tree_child.get());
        }
    }

    /// @brief Computes the specified aggregation operation on an already gathered range of durations.
    ///
    /// @param mode Aggregation operation to perform.
    /// @param gathered_data Durations gathered from all participating ranks.
    /// @param evaluation_node Object where the aggregated and evaluated durations are stored.
    void aggregate_measurements_globally(
        DataAggregationMode                                mode,
        std::vector<double> const&                         gathered_data,
        kamping::measurements::EvaluationTreeNode<double>& evaluation_node
    ) {
        switch (mode) {
            case DataAggregationMode::max: {
                using Operation = internal::Max;
                evaluation_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
            case DataAggregationMode::min: {
                using Operation = internal::Min;
                evaluation_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
            case DataAggregationMode::sum: {
                using Operation = internal::Sum;
                evaluation_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
            case DataAggregationMode::gather: {
                using Operation = internal::Gather;
                evaluation_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
        }
    }
};

/// @brief A basic Timer that uses kamping::Communicator<> as underlying communicator type.
using BasicTimer = Timer<Communicator<>>;

/// @brief Gets a reference to a kamping::measurements::BasicTimer.
///
/// @return A reference to a kamping::measurements::BasicCommunicator.
inline Timer<Communicator<>>& timer() {
    static Timer<Communicator<>> timer;
    return timer;
}

} // namespace kamping::measurements
