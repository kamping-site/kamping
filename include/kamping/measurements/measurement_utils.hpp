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
/// This file contains a (distributed) utility classes and functions that are needed for a distributed timer class.

#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"

namespace kamping::measurements {

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
    min,   ///< The minimum of the measurement data on the participating ranks will be computed.
    max,   ///< The maximum of the measurement data on the participating ranks will be computed.
    sum,   ///< The sum of the measurement data on the participating ranks will be computed.
    gather ///< The measurement data on the participating ranks will be collected in a container.
};
} // namespace kamping::measurements

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

    /// @brief Returns operation's name.
    /// @return Operations name.
    static std::string operation_name() {
        return "sum";
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

/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the same key). A node can have multiple children which represent nested time measurements. The measurements
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
    /// @param mode The kamping::measurements::KeyAggregationMode parameter determines how multiple time measurements
    /// shall be handled. They can either be accumulated (the durations are added together) or appended (the durations
    /// are stored in a list).
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

    /// @brief Access to stored duration(s).
    /// @return Return a reference to duration(s).
    auto& durations() {
        return _durations;
    }

    /// @brief Access to stored duration(s).
    /// @return Return a reference to duration(s).
    auto const& durations() const {
        return _durations;
    }

    /// @brief Access to the data aggregation operations (used during the evaluation).
    /// @return Return a reference to data aggregation operations.
    auto& duration_aggregation_operations() {
        return _duration_aggregation_operations;
    }

    /// @brief Access to the is_active flag indicating whether there is an active time measurement associated with this
    /// node.
    /// @return Return a reference to is_active flag.
    bool& is_active() {
        return _is_active;
    }

private:
    TimePoint                        _start; ///< Point in time at which the current measurement has been started.
    bool                             _is_active{false}; ///< Indicates whether a time measurement is currently active.
    std::vector<Duration>            _durations;        ///< Duration(s) of the node
    std::vector<DataAggregationMode> _duration_aggregation_operations{
        DataAggregationMode::max}; ///< Communicator-wide aggregation operation which will be performed on the
                                   ///< durations. @TODO replace this with a more space efficient variant
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

    /// @brief Resets the root node (i.e. deletes and assigns a new empty node).
    void reset() {
        root         = TimerTreeNode<TimePoint, Duration>{"root"};
        current_node = &root;
    }

    TimerTreeNode<TimePoint, Duration>  root;         ///< Root node of the tree.
    TimerTreeNode<TimePoint, Duration>* current_node; ///< Pointer to the currently active node of the tree.
};
} // namespace kamping::measurements::internal

namespace kamping::measurements {
/// @brief Class representing a node in the timer tree. Each node represents a time measurement (or multiple with
/// the
///  same key). A node can have multiple children which represent nested time measurements. The measurements
///  associated with a node's children are executed while the node's measurement is still active.
///
/// @tparam Duration  Type of a duration.
template <typename Duration>
class EvaluationTreeNode : public internal::TreeNode<EvaluationTreeNode<Duration>> {
public:
    using internal::TreeNode<EvaluationTreeNode<Duration>>::TreeNode;

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
    auto res    = comm.gatherv(send_buf(str));
    auto result = true;
    if (comm.is_root()) {
        auto recv_buf = res.extract_recv_buffer();
        for (std::size_t cur_rank = 0; cur_rank < comm.size(); ++cur_rank) {
            auto              begin = recv_buf.begin() + static_cast<int>(cur_rank * str.size());
            auto              end   = begin + static_cast<int>(str.size());
            const std::string cur_string(begin, end);
            if (cur_string != str) {
                result = false;
                break;
            }
        }
    }
    comm.bcast_single(send_recv_buf(result));
    return result;
}

} // namespace kamping::measurements
