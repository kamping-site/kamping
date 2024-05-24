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
/// This file contains (distributed) utility classes and functions that are needed for a distributed timer class.

#pragma once

#include <memory>
#include <optional>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/measurement_aggregation_definitions.hpp"

namespace kamping::measurements::internal {
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
};

/// @brief Object encapsulating a summation operation on a given range of objects.
struct Sum {
    /// @brief Apply a summation computation of the given range of objects.
    /// @tparam Container Type of container storing the objects.
    /// @param container Container storing objects on which the aggregation operation is applied.
    /// @return std::optional which either contains sum of elements in the container or is empty if container is
    /// empty.
    template <typename Container>
    static auto compute(Container const& container) {
        using T = typename Container::value_type;
        if (container.size() == 0) {
            return std::optional<T>{};
        }
        auto const sum = std::accumulate(container.begin(), container.end(), T{});
        return std::make_optional(sum);
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

/// @brief Object representing a node in a tree. The class is not meant to be used on its own but to encapsulate the
/// basic "tree-node behaviour" (e.g. management of children nodes etc.) for other specialised node classes like
/// TimerTreeNode via CRTP paradigm.
/// @tparam DerivedNode Specialised node type for which the basisc "tree node behaviour" is encapsulated.
template <typename DerivedNode>
class TreeNode {
public:
    /// @brief Constructs node without pointer to parent and empty name.
    TreeNode() : TreeNode(std::string{}) {}

    /// @brief Constructs node without pointer to parent.
    /// @param name Name associated with this node.
    TreeNode(std::string const& name) : TreeNode(name, nullptr) {}

    /// @brief Constructs node pointer to parent.
    /// @param name Name associated with this node.
    /// @param parent Pointer to this node's parent pointer.
    TreeNode(std::string const& name, DerivedNode* parent) : _name{name}, _parent_ptr{parent} {}

    /// @brief Searches the node's children for a node with the given name. If there is no such child a new node is
    /// inserted.
    /// @param name Name of the child which is searched.
    /// @return Reference to the node with the given name.
    auto& find_or_insert(std::string const& name) {
        auto it = _children_map.find(name);
        if (it != _children_map.end()) {
            return *it->second;
        } else {
            auto new_child        = std::make_unique<DerivedNode>(name, static_cast<DerivedNode*>(this));
            auto ptr_to_new_child = new_child.get();
            _children_map[name]   = ptr_to_new_child;
            _children_storage.push_back(std::move(new_child));
            return *ptr_to_new_child;
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

/// @brief Class to store measurement data points associated with a node in a measurement tree, e.g., a timer-tree.
///
/// @tparam T Type of the data point.
/// @tparam default_global_aggregation_mode Default mode to use for global aggregation when not further specified.
template <typename T, GlobalAggregationMode default_global_aggregation_mode>
class NodeMeasurements {
public:
    /// @brief Add the result of a time measurement (i.e. a duration) to the node.
    ///
    /// @param datapoint Data point which is added to the node.
    /// @param mode The kamping::measurements::KeyAggregationMode parameter determines how multiple time measurements
    /// shall be handled. They can either be accumulated (the durations are added together) or appended (the durations
    /// are stored in a list).
    void aggregate_measurements_locally(T const& datapoint, LocalAggregationMode const& mode) {
        switch (mode) {
            case LocalAggregationMode::accumulate:
                if (_datapoints.empty()) {
                    _datapoints.push_back(datapoint);
                } else {
                    _datapoints.back() += datapoint;
                }
                break;
            case LocalAggregationMode::append:
                _datapoints.push_back(datapoint);
        }
    }

    /// @brief Access to stored duration(s).
    /// @return Return a reference to duration(s).
    auto const& measurements() const {
        return _datapoints;
    }

    /// @brief Access to the data aggregation operations (used during the evaluation).
    /// @return Return a reference to data aggregation operations.
    auto& measurements_aggregation_operations() {
        return _datapoint_aggregation_operations;
    }

    /// @brief Access to the data aggregation operations (used during the evaluation).
    /// @return Return a reference to data aggregation operations.
    auto const& measurements_aggregation_operations() const {
        return _datapoint_aggregation_operations;
    }

private:
    std::vector<T>                     _datapoints; ///< Datapoints stored at the node
    std::vector<GlobalAggregationMode> _datapoint_aggregation_operations{
        default_global_aggregation_mode}; ///< Communicator-wide aggregation operation which will be performed on the
                                          ///< measurements. @TODO replace this with a more space efficient variant
};

/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam TimePoint Type of a point in time.
/// @tparam Duration  Type of a duration.
template <typename TimePoint, typename Duration>
class TimerTreeNode : public TreeNode<TimerTreeNode<TimePoint, Duration>>,
                      public NodeMeasurements<Duration, GlobalAggregationMode::max> {
public:
    using TreeNode<TimerTreeNode<TimePoint, Duration>>::TreeNode;

    /// @brief Returns the point in time at which the currently active measurement has been started.
    /// @return Point in time at which the currently active measurement has been started.
    TimePoint startpoint() const {
        return _start;
    }

    /// @brief Sets the point in time at which the currently active measurement has been started.
    /// @param start New start point for currently active measurement.
    void startpoint(TimePoint start) {
        _start = start;
    }

    /// @brief Sets the activity status of this node (i.e. is there a currently active measurement).
    /// @param is_active Activity status to set.
    void is_active(bool is_active) {
        _is_active = is_active;
    }

    /// @brief Getter for activity status.
    /// @return Returns whether this node is associated with a currently active measurement.
    bool is_active() const {
        return _is_active;
    }

private:
    TimePoint _start;            ///< Point in time at which the current measurement has been started.
    bool      _is_active{false}; ///< Indicates whether a time measurement is currently active.
};

/// @brief Class representing a node in the counter tree. Each node represents a measurement (or multiple with
/// the same key). A node can have multiple children which represent nested measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam TimePoint Type of a point in time.
/// @tparam Duration  Type of a duration.
template <typename DataType>
class CounterTreeNode : public TreeNode<CounterTreeNode<DataType>>,
                        public NodeMeasurements<DataType, GlobalAggregationMode::sum> {
public:
    using TreeNode<CounterTreeNode<DataType>>::TreeNode;
};

/// @brief Tree consisting of objects of type \c NodeType. The tree constitutes a hierarchy of measurements
/// such that each node correspond to one (or multiple) measurement(s) with the same name.
/// For timer tree, the measurements corresponding to the node's children are all started and stopped while the node's
/// current time measurement is running.
///
/// @tparam NodeType Underlying node type.
template <typename NodeType>
struct Tree {
    /// @brief Construct a TimerTree consisting only of a root node.
    Tree() : root{"root"}, current_node(&root) {
        root.parent_ptr() = &root;
    }
    /// @brief Resets the root node (i.e. deletes and assigns a new empty node).
    void reset() {
        root         = NodeType{"root"};
        current_node = &root;
    }
    NodeType  root;         ///< Root node of the tree.
    NodeType* current_node; ///< Pointer to the currently active node of the tree.
};

/// @brief Checks that the given string is equal on all ranks in the given communicator.
///
/// @todo this function should be once superseded by a more general Communicator::is_same_on_all_ranks().
/// @tparam Communicator Type of communicator.
/// @param str String which is tested for equality on all ranks.
/// @param comm Communicator on which the equality test is executed.
/// @return Returns whether string is equal on all ranks.
template <typename Communicator>
inline bool is_string_same_on_all_ranks(std::string const& str, Communicator const& comm) {
    auto has_same_size = comm.is_same_on_all_ranks(str.size());
    if (!has_same_size) {
        return false;
    }

    // std::vector<char> name_as_char_vector;
    auto recv_buf = comm.gatherv(send_buf(str));
    auto result   = true;
    if (comm.is_root()) {
        for (std::size_t cur_rank = 0; cur_rank < comm.size(); ++cur_rank) {
            auto              begin = recv_buf.begin() + static_cast<int>(cur_rank * str.size());
            auto              end   = begin + static_cast<int>(str.size());
            std::string const cur_string(begin, end);
            if (cur_string != str) {
                result = false;
                break;
            }
        }
    }
    comm.bcast_single(send_recv_buf(result));
    return result;
}
} // namespace kamping::measurements::internal
