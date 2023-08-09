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

namespace kamping {
namespace timer {
using TimePoint = double;
using Duration  = double;
template <typename T>
using ScalarOrContainer = std::variant<T, std::vector<T>>;
enum class KeyAggregationMode { accumulate, append };
enum class DataAggregationMode { min, max, gather };

template <typename T>
struct Max {
    template <typename Container>
    static T compute(Container const& data) {
        if (data.size() == 0) {
            return T{};
        }
        auto it = std::max_element(data.begin(), data.end());
        return *it;
    }
    static std::string operation_name() {
        return "max";
    }
};

template <typename T>
struct Min {
    template <typename Container>
    static T compute(Container const& data) {
        if (data.size() == 0) {
            return T{};
        }
        auto it = std::min_element(data.begin(), data.end());
        return *it;
    }
    static std::string operation_name() {
        return "min";
    }
};

template <typename T>
struct Gather {
    template <typename Container>
    static Container compute(Container const& data) {
        return data;
    }
    static std::string operation_name() {
        return "gather";
    }
};

template <typename T>
class EvaluatedMeasurementData {
public:
    using EvaluatedDataType = std::unordered_map<std::string, std::vector<ScalarOrContainer<T>>>;
    auto& aggregated_data() {
        return _aggregated_data;
    }
    auto const& aggregated_data() const {
        return _aggregated_data;
    }
    void add(std::string const& aggregation_name, T const& data) {
        _aggregated_data[aggregation_name].emplace_back(data);
    }
    void add(const std::string aggregation_name, std::vector<T> const& data) {
        _aggregated_data[aggregation_name].emplace_back(data);
    }

private:
    EvaluatedDataType _aggregated_data;
};

template <typename TimePoint, typename Duration>
class TimerTreeNode {
public:
    TimerTreeNode() : _name{}, _parent_ptr{nullptr} {}
    TimerTreeNode(std::string const& name) : _name{name}, _parent_ptr{nullptr} {}
    TimerTreeNode(std::string const& name, TimerTreeNode<TimePoint, Duration>* parent)
        : _name{name},
          _parent_ptr{parent} {}
    auto find_or_insert(std::string const& name) {
        auto it = _children_map.find(name);
        if (it != _children_map.end()) {
            return it->second;
        } else {
            auto new_child        = std::make_unique<TimerTreeNode<TimePoint, Duration>>(name, this);
            auto ptr_to_new_child = new_child.get();
            _children_map[name]   = ptr_to_new_child;
            _children_storage.push_back(std::move(new_child));
            return ptr_to_new_child;
        }
    }
    auto find(std::string const& name) {
        auto it = _children_map.find(name);
        return it->second;
    }
    auto& startpoint() {
        return _start;
    }
    void aggregate(Duration const& data_item, KeyAggregationMode const& mode = KeyAggregationMode::accumulate) {
        ++num_calls;
        switch (mode) {
            case KeyAggregationMode::accumulate:
                if (_data.empty()) {
                    _data.push_back(data_item);
                } else {
                    _data.back() += data_item;
                }
                break;
            case KeyAggregationMode::append:
                _data.push_back(data_item);
        }
    }
    auto& parent_ptr() {
        return _parent_ptr;
    }

    void evaluate(Communicator<> const& comm) {
        _evaluated_data = std::make_unique<EvaluatedMeasurementData<Duration>>();
        // TODO some kasserts for same name and number of items
        for (auto const& item: _data) {
            auto recv_buf = comm.gather(send_buf(item)).extract_recv_buffer();
            if (!comm.is_root()) {
                continue;
            }

            for (auto const& aggregation_mode: _data_aggregation) {
                evaluate(aggregation_mode, recv_buf);
            }
        }
        for (auto& child: _children_storage) {
            child->evaluate(comm);
        }
    }

    auto const& children() const {
        return _children_storage;
    }
    auto& data_aggregation() {
        return _data_aggregation;
    }
    auto get_print_data() const {
        typename EvaluatedMeasurementData<Duration>::EvaluatedDataType evaluation_data;
        if (_evaluated_data) {
            evaluation_data = _evaluated_data->aggregated_data();
        }
        return std::make_pair(_name, evaluation_data);
    }

private:
    std::string                                         _name;
    TimePoint                                           _start;
    std::vector<Duration>                               _data;
    std::vector<DataAggregationMode>                    _data_aggregation{DataAggregationMode::max};
    TimerTreeNode*                                      _parent_ptr;
    std::unordered_map<std::string, TimerTreeNode*>     _children_map;
    std::vector<std::unique_ptr<TimerTreeNode>>         _children_storage;
    std::unique_ptr<EvaluatedMeasurementData<Duration>> _evaluated_data;
    size_t                                              num_calls{0};

    void evaluate(DataAggregationMode mode, std::vector<Duration> const& gathered_data) {
        switch (mode) {
            case DataAggregationMode::max:
                _evaluated_data->add(Max<Duration>::operation_name(), Max<Duration>::compute(gathered_data));
                break;
            case DataAggregationMode::min:
                _evaluated_data->add(Min<Duration>::operation_name(), Min<Duration>::compute(gathered_data));
                break;
            case DataAggregationMode::gather:
                _evaluated_data->add(Gather<Duration>::operation_name(), Gather<Duration>::compute(gathered_data));
                break;
        }
    }
};

template <typename T>
struct InternalPrinter {
    void operator()(std::vector<T> const& vec) const {
        std::cout << "[";
        bool is_first = true;
        for (auto const& elem: vec) {
            if (!is_first) {
                std::cout << ", ";
            }
            is_first = false;
            std::cout << std::fixed << elem;
        }
        std::cout << "]";
    }
    void operator()(T const& scalar) const {
        std::cout << std::fixed << scalar;
    }
};

struct SimpleJsonPrinter {
    template <typename TimePoint, typename Duration>
    void print(TimerTreeNode<TimePoint, Duration> const& node, std::size_t indentation = 0) {
        auto [name, evaluation_data] = node.get_print_data();
        std::cout << std::string(indentation, ' ') << "\"" << name << "\": {" << std::endl;

        InternalPrinter<Duration> internal_printer;
        std::cout << std::string(indentation + 2, ' ') << "\"statistics\": {" << std::endl;
        if (!evaluation_data.empty()) {
            bool is_first_outer = true;
            for (auto const& [op, data]: evaluation_data) {
                if (!is_first_outer) {
                    std::cout << "," << std::endl;
                }
                is_first_outer = false;
                std::cout << std::string(indentation + 4, ' ') << "\"" << op << "\""
                          << ": [";
                bool is_first = true;
                for (auto const& data_item: data) {
                    if (!is_first) {
                        std::cout << ", ";
                    }
                    is_first = false;
                    std::visit(internal_printer, data_item);
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        std::cout << std::string(indentation + 2, ' ') << "}";
        if (node.children().size() != 0) {
            std::cout << ",";
        }
        std::cout << std::endl;

        bool is_first = true;
        for (auto const& children: node.children()) {
            if (!is_first) {
                std::cout << "," << std::endl;
            }
            is_first = false;
            print(*children, indentation + 2);
        }
        if (node.children().size() != 0) {
            std::cout << std::endl;
        }
        std::cout << std::string(indentation, ' ') << "}";
    }
};

template <typename TimePoint, typename Duration>
struct MeasurementTree {
    MeasurementTree() : root{"root"}, current_node(&root) {
        root.parent_ptr() = &root;
    }
    TimerTreeNode<TimePoint, Duration>  root;
    TimerTreeNode<TimePoint, Duration>* current_node;
};

class Timer {
public:
    void synchronize_and_start(std::string const& key) {
        bool const use_barrier = true;
        start_impl(key, use_barrier);
    };
    void start(std::string const& key) {
        bool const use_barrier = false;
        start_impl(key, use_barrier);
    };

    void stop(std::vector<DataAggregationMode> const& data_aggreation_modi = std::vector<DataAggregationMode>{}) {
        stop_impl(KeyAggregationMode::accumulate, data_aggreation_modi);
    };
    void
    stop_and_append(std::vector<DataAggregationMode> const& data_aggreation_modi = std::vector<DataAggregationMode>{}) {
        stop_impl(KeyAggregationMode::append, data_aggreation_modi);
    };
    void stop_and_accumulate(
        std::vector<DataAggregationMode> const& data_aggreation_modi = std::vector<DataAggregationMode>{}
    ) {
        stop_impl(KeyAggregationMode::accumulate, data_aggreation_modi);
    };
    template <typename Printer>
    void print(Printer&& printer) {
        if (comm_world().is_root()) {
            printer.print(_timer.root);
        }
    }
    void evaluate() {
        _timer.root.evaluate(comm_world());
    }

private:
    MeasurementTree<double, double> _timer;

    void start_impl(std::string const& key, bool use_barrier) {
        auto node = _timer.current_node->find_or_insert(key);
        if (use_barrier) {
            comm_world().barrier();
        }
        node->startpoint()  = Environment<>::wtime();
        _timer.current_node = node;
    };

    void stop_impl(KeyAggregationMode mode, std::vector<DataAggregationMode> const& data_aggreation_modi) {
        auto endpoint   = Environment<>::wtime();
        auto startpoint = _timer.current_node->startpoint();
        _timer.current_node->aggregate(endpoint - startpoint, mode);
        if (data_aggreation_modi.size() != 0) {
            _timer.current_node->data_aggregation() = data_aggreation_modi;
        }
        _timer.current_node = _timer.current_node->parent_ptr();
    };
};
}; // namespace timer

} // namespace kamping
