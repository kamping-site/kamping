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
#include "kamping/measurements/aggregated_tree_node.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"

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
///
/// timer.aggregate_and_print(Printer{}); // aggregates measurements across all participating ranks and print results
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
/// ## Aggregation operations ##
///
/// There are two types of aggregation operations:
///
/// 1) Local aggregation operations - It is possible to execute measurements with the same key multiple times. Local
/// aggregation operations specify how these repeated time measurements will be stored. Currently, there are two
/// options - stop_and_add() and stop_and_append(). See the following examples:
///
/// \code
/// timer.start("foo");
/// timer.stop_and_add();
/// timer.start("foo");
/// timer.stop_and_add();
/// \endcode
/// The result of this program is one stored duration for the key "foo" which is the sum of the durations for the two
/// measurements with key "foo".
///
/// The other option to handle repeated measurements with the same key is stop_and_append():
/// \code
/// timer.start("bar");
/// timer.stop_and_append();
/// timer.start("bar");
/// timer.stop_and_append();
/// \endcode
///
/// This program results in a list with two durations for the key "bar". We call the  number of durations
/// stored for a key its \c dimension.
///
/// 2) Global (communicator-wide) aggregation operations specify how the stored duration(s) for a specific measurement
/// key shall be aggregated. The default operation (if no operations are specified via the stop() method) is to take the
/// maximum duration over all ranks in the communicator. If there are multiple durations for a certain key, the
/// aggregation operation is applied element-wise and the result remains a list with the same size as the number of
/// input durations. The communicator-wide aggregation operation are applied in the evaluation phase started with
/// calls to aggregate() or aggregate_and_print().
///
///
/// The timer hierarchy that is implicitly created by start() and stop() must be the same on all ranks in the given
/// communicator. The number of time measurements with a specific key may vary as long as the number of stored duration
/// are the same:
/// Consider for example a communicator of size two and rank 0 and 1 both measure the time of a function \c
/// foo() each time it is called using the key \c "computation". This is only valid if either each rank calls foo()
/// exactly the same number of times and local aggregation happens using the mode \c append, or if the measured time is
/// locally aggregated using add, i.e. the value(s) stored for a single key at the same hierarchy level must have
/// matching dimensions.
/// Furthermore, global communicator-wide duration aggregation operations specified on ranks other than the root rank
/// are ignored.
///
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
    Timer(CommunicatorType const& comm) : _timer_tree{}, _comm{comm}, _is_timer_enabled{true} {}

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
    /// @return AggregatedTree object which encapsulated the aggregated data in a tree structure representing the
    /// measurements.
    auto aggregate() {
        AggregatedTree<Duration> aggregated_tree(_timer_tree.root, _comm);
        return aggregated_tree;
    }

    /// @brief Clears all stored measurements.
    void clear() {
        _timer_tree.reset();
    }

    /// @brief (Re-)Enable start/stop operations.
    void enable() {
        _is_timer_enabled = true;
    }
    /// @brief Disable start/stop operations, i.e., start()/stop() operations do not have any effect.
    void disable() {
        _is_timer_enabled = false;
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
        auto const aggregated_tree = aggregate();
        if (_comm.is_root()) {
            printer.print(aggregated_tree.root());
        }
    }

private:
    internal::Tree<internal::TimerTreeNode<double, Duration>>
                            _timer_tree;       ///< Timer tree used to represent the hierarchical time measurements.
    CommunicatorType const& _comm;             ///< Communicator in which the time measurements take place.
    bool                    _is_timer_enabled; ///< Flag indicating whether start/stop operations are enabled.

    /// @brief Starts a time measurement.
    void start_impl(std::string const& key, bool use_barrier) {
        if (!_is_timer_enabled) {
            return;
        }
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
    /// ranks when aggregate() is called.
    void stop_impl(
        LocalAggregationMode local_aggregation_mode, std::vector<GlobalAggregationMode> const& global_aggregation_modes
    ) {
        if (!_is_timer_enabled) {
            return;
        }
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
            _timer_tree.current_node->measurements_aggregation_operations() = global_aggregation_modes;
        }
        _timer_tree.current_node = _timer_tree.current_node->parent_ptr();
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
