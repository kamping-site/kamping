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

#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

/// @file
/// This file contains a (distributed) timer class (and helpers)

namespace kamping::timer {

///@brief Either a scalar or vector of type \c T.
///@tparam T Type.
template <typename T>
using ScalarOrContainer = std::variant<T, std::vector<T>>;

/// @brief Enum to specify how time measurements with same key shall be aggregated.
enum class KeyAggregationMode {
    accumulate, ///< Tag used to indicate that data associated with identical keys will be accumulated into a scalar.
    append      ///< Tag used to indicate that data with identical keys will not be accumulated and stored in a list.
};

/// @brief Enum to specify how time duration shall be aggregated across the participating ranks.
enum class DataAggregationMode {
    min,   ///< The minimum of the durations on the participating ranks will be computed.
    max,   ///< The maximum of the durations on the participating ranks will be computed.
    gather ///< The duration data on the participating ranks will be collected in a container.
};

/// @brief Object encapsulating a maximum operation on a given range of objects.
struct Max {
    /// @brief Apply a maximum computation of the given range of objects.
    /// @tparam Container Type of container storing the objects.
    /// @param container Container storing objects on which the aggregation operation is applied.
    /// @return std::optional which either contains maximum of elements in the container or is empty if container is
    /// empty.
    template <typename Container>
    static auto compute(Container const& container) {
        using T = typename Container::value_type;
        if (container.size() == 0) {
            return std::optional<T>{};
        }
        auto it = std::max_element(container.begin(), container.end());
        return std::make_optional(*it);
    }
    /// @brief Returns operation's name.
    /// @return Operations name.
    static std::string operation_name() {
        return "max";
    }
};

/// @brief Object encapsulating a minimum operation on a given range of objects.
struct Min {
    /// @brief Apply a minimum computation of the given range of objects.
    /// @tparam Container Type of container storing the objects.
    /// @param container Container storing objects on which the aggregation operation is applied.
    /// @return std::optional which either contains minimum of elements in the container or is empty if container is
    /// empty.
    template <typename Container>
    static auto compute(Container const& container) {
        using T = typename Container::value_type;
        if (container.size() == 0) {
            return std::optional<T>{};
        }
        auto it = std::min_element(container.begin(), container.end());
        return std::make_optional(*it);
    }

    /// @brief Returns operation's name.
    /// @return Operations name.
    static std::string operation_name() {
        return "min";
    }
};

/// @brief Object encapsulating a gather operation on a given range of objects.
struct Gather {
    /// @brief Forwards input container.
    /// @tparam Container Type of container storing the objects.
    /// @param container Container storing objects which is forward.
    /// @return Forwarded container.
    template <typename Container>
    static decltype(auto) compute(Container&& container) {
        return std::forward<Container>(container);
    }

    /// @brief Returns operation's name.
    /// @return Operations name.
    static std::string operation_name() {
        return "gather";
    }
};

/// @brief Stores aggregated data.
///
/// @tparam T Underlying datatype which is aggregated.
template <typename T>
class AggregatedMeasurementDataStorage {
public:
    ///@brief Type into which the aggregated data is stored together with the applied aggregation operation.
    using StorageType = std::unordered_map<std::string, std::vector<ScalarOrContainer<T>>>;

    /// @brief Access to stored aggregated data.
    /// @return Reference to aggregated data.
    auto& aggregated_data() {
        return _aggregated_data;
    }

    /// @brief Access to stored aggregated data.
    /// @return Reference to aggregated data.
    auto const& aggregated_data() const {
        return _aggregated_data;
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_operation Applied aggregation operations.
    /// @param data Scalar resulted from applying the given aggregation operation.
    void add(std::string const& aggregation_operation, std::optional<T> data) {
        if (data) {
            _aggregated_data[aggregation_operation].emplace_back(data.value());
        }
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_operation Applied aggregation operations.
    /// @param data Vector of Scalars resulted from applying the given aggregation operation.
    void add(const std::string aggregation_operation, std::vector<T> const& data) {
        _aggregated_data[aggregation_operation].emplace_back(data);
    }

private:
    StorageType _aggregated_data; ///< Storage of the aggregated data.
};

template <typename T>
class TD;

/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the
///  same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
template <typename DerivedNode>
class TreeNode {
public:
    /// @brief Construct node without pointer to parent and empty name
    TreeNode() : _name{}, _parent_ptr{nullptr} {}

    /// @brief Construct node without pointer to parent.
    /// @param name Name associated with this node.
    TreeNode(std::string const& name) : _name{name}, _parent_ptr{nullptr} {}

    /// @brief Construct node  pointer to parent.
    /// @param name Name associated with this node.
    /// @param parent Pointer to this node's parent pointer.
    TreeNode(std::string const& name, DerivedNode* parent) : _name{name}, _parent_ptr{parent} {}

    /// @brief Searches the node's children for a node with the given name. If there is no such child a new node is
    /// inserted.
    /// @param name Name of the child which is searched.
    /// @return Pointer to the node with the given name.
    auto find_or_insert(std::string const& name) {
        auto it = _children_map.find(name);
        if (it != _children_map.end()) {
            return it->second;
        } else {
            auto new_child        = std::make_unique<DerivedNode>(name, static_cast<DerivedNode*>(this));
            auto ptr_to_new_child = new_child.get();
            _children_map[name]   = ptr_to_new_child;
            _children_storage.push_back(std::move(new_child));
            // TD<decltype(ptr_to_new_child)> td;
            return ptr_to_new_child;
        }
    }

    /// @brief Access to the parent pointer.
    /// @return Reference to the parent pointer.
    auto& parent_ptr() {
        return _parent_ptr;
    }

    /// @brief Access to the node's children.
    /// @return Return a reference to the node's children.
    auto const& children() const {
        return _children_storage;
    }

    /// @brief Access to the node's children.
    /// @return Return a reference to the node's children.
    auto const& name() const {
        return _name;
    }

private:
    std::string  _name;       ///< Name of the node.
    DerivedNode* _parent_ptr; ///< Pointer to the node's parent.
    std::unordered_map<std::string, DerivedNode*>
                                              _children_map; ///< Map (used for faster lookup) to the node's children.
    std::vector<std::unique_ptr<DerivedNode>> _children_storage; ///< Owns the node's children.
};

/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the
///  same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam TimePoint Type of a point in time.
/// @tparam Duration  Type of a duration.
template <typename TimePoint, typename Duration>
class TimerTreeNode : public TreeNode<TimerTreeNode<TimePoint, Duration>> {
public:
    using TreeNode<TimerTreeNode<TimePoint, Duration>>::TreeNode;
    /// @brief Access to the point in time at which the currently active measurement has been started.
    /// @return Reference to start point.
    auto& startpoint() {
        return _start;
    }

    /// @brief Add the result of a time measurement (i.e. a duration) to the node.
    ///
    /// @param duration Duration which is added to the node.
    /// @param mode The kamping::timer::KeyAggregationMode parameter determines how multiple time measurements shall
    /// be handled. They can either be accumulated (the durations are added together) or appended (the durations are
    /// stored in a list).
    void aggregate_measurements_locally(Duration const& duration, KeyAggregationMode const& mode) {
        switch (mode) {
            case KeyAggregationMode::accumulate:
                if (_durations.empty()) {
                    _durations.push_back(duration);
                } else {
                    _durations.back() += duration;
                }
                break;
            case KeyAggregationMode::append:
                _durations.push_back(duration);
        }
    }

    /// @brief Access to the data aggregation operations (used during the evaluation).
    /// @return Return a reference to data aggregation operations.
    auto& data_aggregation_operations() {
        return _duration_aggregation_operations;
    }

public:
    TimePoint                        _start;     ///< Point in time at which the current measurement has been started.
    std::vector<Duration>            _durations; ///< Duration(s) of the node
    std::vector<DataAggregationMode> _duration_aggregation_operations{
        DataAggregationMode::max}; ///< Communicator-wide aggregation operation which will be performed on the
                                   ///< durations.
};

/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the
///  same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam Duration  Type of a duration.
template <typename Duration>
class EvaluationTreeNode : public TreeNode<EvaluationTreeNode<Duration>> {
public:
    using TreeNode<EvaluationTreeNode<Duration>>::TreeNode;

    ///@brief Type into which the aggregated data is stored together with the applied aggregation operation.
    using StorageType = std::unordered_map<std::string, std::vector<ScalarOrContainer<Duration>>>;

    /// @brief Access to stored aggregated data.
    /// @return Reference to aggregated data.
    auto const& aggregated_data() const {
        return _aggregated_data;
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_operation Applied aggregation operations.
    /// @param data Scalar resulted from applying the given aggregation operation.
    void add(std::string const& aggregation_operation, std::optional<Duration> data) {
        if (data) {
            _aggregated_data[aggregation_operation].emplace_back(data.value());
        }
    }

    /// @brief Add scalar of type T to aggregated data storage together with the name of the  applied aggregation
    /// operation.
    /// @param aggregation_operation Applied aggregation operations.
    /// @param data Vector of Scalars resulted from applying the given aggregation operation.
    void add(const std::string aggregation_operation, std::vector<Duration> const& data) {
        _aggregated_data[aggregation_operation].emplace_back(data);
    }

public:
    StorageType _aggregated_data; ///< Storage of the aggregated data.
};

/// @brief Tree consisting of objects of type TimerTreeNode. The tree constitutes a hierarchy of time measurements
/// such that each node correspond to one (or multiple) time measurement(s) with the same name and the time
/// measurements corresponding to the node's children are all started and stopped while the node's current time
/// measurement is running.
template <typename TimePoint, typename Duration>
struct TimerTree {
    /// @brief Construct a TimerTree consisting only of a root node.
    TimerTree() : root{"root"}, current_node(&root) {
        root.parent_ptr() = &root;
    }
    TimerTreeNode<TimePoint, Duration>  root;         ///< Root node of the tree.
    TimerTreeNode<TimePoint, Duration>* current_node; ///< Pointer to the currently active node of the tree.
};

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
    };

    /// @brief Starts the measuremt with the given key.
    /// @param key Key with which the started time measurement is associated.
    void start(std::string const& key) {
        bool const use_barrier = false;
        start_impl(key, use_barrier);
    };

    /// @brief Stops the currently active measurement and store the result.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop(std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}) {
        stop_impl(KeyAggregationMode::accumulate, duration_aggregation_modi);
    };

    /// @brief Stops the currently active measurement and store the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is added to
    /// the last measured duration at this hierarchy level with the same key.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop_and_accumulate(
        std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}
    ) {
        stop_impl(KeyAggregationMode::accumulate, duration_aggregation_modi);
    };

    /// @brief Stops the currently active measurement and store the result. If the key associated with the
    /// measurement that is stopped has already been used at the current hierarchy level the duration is append to a
    /// list of previously measured durations at this hierarchy level with the same key.
    /// @param duration_aggregation_modi Specify how the measurement duration is aggregated over all participationg
    /// PEs when Timer::aggregate() is called.
    void stop_and_append(
        std::vector<DataAggregationMode> const& duration_aggregation_modi = std::vector<DataAggregationMode>{}
    ) {
        stop_impl(KeyAggregationMode::append, duration_aggregation_modi);
    };

    /// @brief Evaluates the time measurements represented by the timer tree for which the current node is the root
    /// node globally. The measured durations are aggregated over all participating PEs and the result is stored at
    /// the root rank of the given communicator. The used aggregation operations can be specified via
    /// TimerTreeNode::data_aggregation_operations().
    /// The durations are aggregated node by node.
    ///
    /// @return Root of the evaluation tree which encapsulated the aggregated data in a tree structure representing the
    /// measurements.
    auto evaluate() {
        EvaluationTreeNode<double> root;
        evaluate(root, _timer_tree.root);
        return root;
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
    TimerTree<double, double> _timer_tree; ///< Timer tree used to represent the hiearchical time measurements.
    CommunicatorType          _comm;       ///< Communicator in which the time measurements take place.

    /// @brief Starts a time measurement.
    void start_impl(std::string const& key, bool use_barrier) {
        auto node = _timer_tree.current_node->find_or_insert(key);
        if (use_barrier) {
            comm_world().barrier();
        }
        node->startpoint()       = Environment<>::wtime();
        _timer_tree.current_node = node;
    };

    /// @brief Stops a time measurement and stores result.
    void
    stop_impl(KeyAggregationMode key_aggregation_mode, std::vector<DataAggregationMode> const& data_aggreation_modi) {
        auto endpoint   = Environment<>::wtime();
        auto startpoint = _timer_tree.current_node->startpoint();
        _timer_tree.current_node->aggregate_measurements_locally(endpoint - startpoint, key_aggregation_mode);
        if (data_aggreation_modi.size() != 0) {
            _timer_tree.current_node->data_aggregation_operations() = data_aggreation_modi;
        }
        _timer_tree.current_node = _timer_tree.current_node->parent_ptr();
    };

    void evaluate(EvaluationTreeNode<double>& aggregation_node, TimerTreeNode<double, double>& timer_node) {
        for (auto& child: timer_node.children()) {
            auto aggregation_child = aggregation_node.find_or_insert(child->name());
            // TODO some kasserts for same name and number of items
            for (auto const& item: child->_durations) {
                auto recv_buf = _comm.gather(send_buf(item)).extract_recv_buffer();
                if (!_comm.is_root()) {
                    continue;
                }

                for (auto const& aggregation_mode: child->_duration_aggregation_operations) {
                    aggregate_measurements_globally(aggregation_mode, recv_buf, *aggregation_child);
                }
            }
            evaluate(*aggregation_child, *child.get());
        }
    }

    void aggregate_measurements_globally(
        DataAggregationMode                         mode,
        std::vector<double> const&                  gathered_data,
        kamping::timer::EvaluationTreeNode<double>& agg_node
    ) {
        switch (mode) {
            case DataAggregationMode::max: {
                using Operation = Max;
                agg_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
            case DataAggregationMode::min: {
                using Operation = Min;
                agg_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
            case DataAggregationMode::gather: {
                using Operation = Gather;
                agg_node.add(Operation::operation_name(), Operation::compute(gathered_data));
                break;
            }
        }
    }
};
} // namespace kamping::timer
