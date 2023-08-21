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
/// ## Timer hierarchy: ##
/// The timer is able to execute time measurements in a hierarchical manner.
/// Time measurements are executed with a matching pair of calls to Timer::start() and Timer::stop().
/// Each call to start() enters a new level in the hierarchy which is left again with the corresponding call to
/// stop(). Time measurements can be nested and the parent time measurement remains active while its child time
/// measurement take place.
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
/// The timer hierarchy that is implicitly created by start() and stop() must be identical on all ranks in the given
/// communicator. The number of time measurements with a specific key may vary as long as the stored duration are
/// identical (e.g. a measurement with "send_data" is execute one time on rank 0 and two times on rank 1 but the
/// durations are accumulated on rank 1 (with stop_and_add()) resulting in the same number of stored durations as
/// on rank 0).
/// Specified communicator-wide duration aggregation operations on ranks other than the root rank are ignored.
///
/// ## Aggregation operations ##
///
/// There are two types of aggregation operations:
///
/// 1) Local aggregations that specifies how repeated time measurements with the same key will be stored.
/// \code
/// timer.start("measurementA");
/// timer.stop_and_add();
/// timer.start("measurementA");
/// timer.stop_and_add();
/// \endcode
/// results in one stored duration for the key "measurementA" which is the sum of the two measured durations for
/// "measurementA", whereas
///
/// \code
/// timer.start("measurementB");
/// timer.stop_and_append();
/// timer.start("measurementB");
/// timer.stop_and_append();
/// \endcode
///
/// results in a list with two durations for the key "measurementB".
///
/// 2) Global (communicator-wide) aggregation operations specify how the stored duration(s) for a specific measurement
/// key shall be aggregated. The default operation (if no operations are specified via the stop() method) is to take the
/// maximum duration over all ranks in the communicator. If there are multiple durations for a certain key, the
/// aggregation operation is applied element-wise and the result remains a list with the same size as the number of
/// input durations. The communicator-wide aggregation operation are applied in the evaluation phase started with
/// calls to evaluate() or evaluate_and_print().
///
/// @see The usage example for Timer provides some more information on how the class can be used.
///
/// @tparam CommunicatorType Communicator in which the time measurements are executed.
template <typename CommunicatorType = Communicator<>>
class Timer {
public:
    using Duration = double; ///< Type of durations. @todo make this customizable.
    /// @brief Constructs a timer using the \c MPI_COMM_WORLD communicator.
    Timer() : Timer{comm_world()} {}

    /// @brief Constructs a timer using a given communicator.
    ///
    /// @param comm Communicator in which the time measurements are executed.
    Timer(CommunicatorType const& comm) : _timer_tree{}, _comm{comm} {}

    /// @brief Synchronizes all ranks in the underlying communicator via a barrier and then start the measurement with
    /// the given key.
    /// @param key Key with which the started time measurement is associated. Note that the user is responsible for
    /// providing keys, which are valid in the used output format during printing.
    void synchronize_and_start(std::string const& key) {
        bool const use_barrier = true;
        start_impl(key, use_barrier);
    }

    /// @brief Starts the measurement with the given key.
    /// @param key Key with which the started time measurement is associated. Note that the user is responsible for
    /// providing keys, which are valid in the used output format during printing.
    void start(std::string const& key) {
        bool const use_barrier = false;
        start_impl(key, use_barrier);
    }

    /// @brief Stops the currently active measurement and stores the result.
    /// @todo Add a toggle option to the Timer class to set the default key aggregation behaviour for stop().
    /// @todo Replace the global_aggregation_modes parameter with a more efficient and ergonomic solution using
    /// variadic templates.
    /// @param global_aggregation_modes Specifies how the measurement duration is aggregated over all participating
    /// ranks when aggregate() is called.
    void
    stop(std::vector<GlobalAggregationMode> const& global_aggregation_modes = std::vector<GlobalAggregationMode>{}) {
        stop_impl(LocalAggregationMode::accumulate, global_aggregation_modes);
    }

    /// @brief Stops the currently active measurement and stores the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is added to
    /// the last measured duration at this hierarchy level with the same key.
    /// @param global_aggregation_modes Specifies how the measurement duration is aggregated over all participating
    /// ranks when aggregate() is called.
    void stop_and_add(
        std::vector<GlobalAggregationMode> const& global_aggregation_modes = std::vector<GlobalAggregationMode>{}
    ) {
        stop_impl(LocalAggregationMode::accumulate, global_aggregation_modes);
    }

    /// @brief Stops the currently active measurement and stores the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is appended to a
    /// list of previously measured durations at this hierarchy level with the same key.
    /// @param global_aggregation_modes Specifies how the measurement duration is aggregated over all participating
    /// ranks when aggregate() is called.
    void stop_and_append(
        std::vector<GlobalAggregationMode> const& global_aggregation_modes = std::vector<GlobalAggregationMode>{}
    ) {
        stop_impl(LocalAggregationMode::append, global_aggregation_modes);
    }

    /// @brief Evaluates the time measurements represented by the timer tree for which the current node is the root
    /// node globally. The measured durations are aggregated over all participating ranks and the result is stored at
    /// the root rank of the given communicator. The used aggregation operations can be specified via
    /// TimerTreeNode::data_aggregation_operations().
    /// The durations are aggregated node by node.
    ///
    /// @return Root of the evaluation tree which encapsulated the aggregated data in a tree structure representing the
    /// measurements.
    auto aggregate() {
        AggregatedTreeNode<Duration> root("root");
        aggregate(root, _timer_tree.root);
        return root;
    }

    /// @brief Clears all stored measurements.
    void clear() {
        _timer_tree.reset();
    }

    /// @brief Aggregates and outputs the executed measurements. The output is done via the print()
    /// method of a given Printer object.
    ///
    /// The print() method must accept an object of type AggregatedTreeNode and receives the root of the evaluated timer
    /// tree as parameter. The print() method is only called on the root rank of the communicator. See
    /// AggregatedTreeNode for the accessible data. The AggregatedTreeNode::children() member function can be used to
    /// navigate the nested measurement structure.
    ///
    /// @tparam Printer Type of printer which is used to output the aggregated timing data. Printer must possess a
    /// member print() which accepts a AggregatedTreeNode as parameter.
    /// @param printer Printer object used to output the aggregated timing data.
    template <typename Printer>
    void aggregate_and_print(Printer&& printer) {
        auto evaluation_tree_root = aggregate();
        if (_comm.is_root()) {
            printer.print(evaluation_tree_root);
        }
    }

private:
    internal::TimerTree<double, Duration>
                            _timer_tree; ///< Timer tree used to represent the hierarchical time measurements.
    CommunicatorType const& _comm;       ///< Communicator in which the time measurements take place.

    /// @brief Starts a time measurement.
    void start_impl(std::string const& key, bool use_barrier) {
        auto& node = _timer_tree.current_node->find_or_insert(key);
        node.is_active(true);
        if (use_barrier) {
            _comm.barrier();
        }
        auto start_point = Environment<>::wtime();
        node.startpoint(start_point);
        _timer_tree.current_node = &node;
    }

    /// @brief Stops the currently active measurement and stores the result.
    /// @param local_aggregation_mode Specifies how the measurement duration is locally aggregated when there are
    /// multiple measurements at the same level with identical key.
    /// @param global_aggregation_modes Specifies how the measurement duration is aggregated over all participating
    /// ranks when evaluate() is called.
    void stop_impl(
        LocalAggregationMode local_aggregation_mode, std::vector<GlobalAggregationMode> const& global_aggregation_modes
    ) {
        auto endpoint = Environment<>::wtime();
        KASSERT(
            _timer_tree.current_node->is_active(),
            "There is no corresponding call to start() associated with this call to stop()",
            assert::light
        );
        _timer_tree.current_node->is_active(false);
        auto startpoint = _timer_tree.current_node->startpoint();
        _timer_tree.current_node->aggregate_measurements_locally(endpoint - startpoint, local_aggregation_mode);
        if (!global_aggregation_modes.empty()) {
            _timer_tree.current_node->duration_aggregation_operations() = global_aggregation_modes;
        }
        _timer_tree.current_node = _timer_tree.current_node->parent_ptr();
    }

    /// @brief Traverses and evaluates the given TimerTreeNode and stores the result in the corresponding
    /// AggregatedTreeNode
    ///
    /// param evaluation_tree_node Node where the aggregated durations are stored.
    /// param timer_tree_node Node where the raw durations are stored.
    void aggregate(
        AggregatedTreeNode<Duration>& evaluation_tree_node, internal::TimerTreeNode<double, Duration>& timer_tree_node
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

        // gather all durations at once as gathering all durations individually may deteriorate
        // the performance of the evaluation operation significantly.
        auto       recv_buf      = _comm.gatherv(send_buf(timer_tree_node.durations())).extract_recv_buffer();
        auto const num_durations = timer_tree_node.durations().size();
        for (size_t duration_idx = 0; duration_idx < num_durations; ++duration_idx) {
            if (!_comm.is_root()) {
                continue;
            }
            std::vector<Duration> cur_durations;
            cur_durations.reserve(_comm.size());
            // gather the durations belonging to the same measurement
            for (size_t rank = 0; rank < _comm.size(); ++rank) {
                cur_durations.push_back(recv_buf[duration_idx + rank * num_durations]);
            }

            for (auto const& aggregation_mode: timer_tree_node.duration_aggregation_operations()) {
                aggregate_measurements_globally(aggregation_mode, cur_durations, evaluation_tree_node);
            }
        }
        for (auto& timer_tree_child: timer_tree_node.children()) {
            auto& evaluation_tree_child = evaluation_tree_node.find_or_insert(timer_tree_child->name());
            aggregate(evaluation_tree_child, *timer_tree_child.get());
        }
    }

    /// @brief Computes the specified aggregation operation on an already gathered range of durations.
    ///
    /// @param mode Aggregation operation to perform.
    /// @param gathered_data Durations gathered from all participating ranks.
    /// @param evaluation_node Object where the aggregated and evaluated durations are stored.
    void aggregate_measurements_globally(
        GlobalAggregationMode                              mode,
        std::vector<Duration> const&                       gathered_data,
        kamping::measurements::AggregatedTreeNode<double>& evaluation_node
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

/// @brief Gets a reference to a kamping::measurements::BasicTimer.
///
/// @return A reference to a kamping::measurements::BasicCommunicator.
inline Timer<Communicator<>>& timer() {
    static Timer<Communicator<>> timer;
    return timer;
}

} // namespace kamping::measurements
