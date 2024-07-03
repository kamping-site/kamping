// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <kamping/communicator.hpp>
#include <kamping/span.hpp>
#include <kamping/topology_communicator.hpp>

namespace kamping {

class CommunicationGraphView {
public:
    using SpanType = kamping::Span<int const>;

    template <typename ContiguousRange>
    CommunicationGraphView(ContiguousRange const& in_ranks, ContiguousRange const& out_ranks)
        : _in_ranks{in_ranks.data(), in_ranks.size()},
          _out_ranks{out_ranks.data(), out_ranks.size()} {}

    template <typename ContiguousRange>
    CommunicationGraphView(
        ContiguousRange const& in_ranks,
        ContiguousRange const& out_ranks,
        ContiguousRange const& in_weights,
        ContiguousRange const& out_weights
    )
        : _in_ranks{in_ranks.data(), in_ranks.size()},
          _out_ranks{out_ranks.data(), out_ranks.size()},
          _in_weights{SpanType{in_weights.data(), in_weights.size()}},
          _out_weights{SpanType{out_weights.data(), out_weights.size()}} {}

    size_t in_degree() const {
        return _in_ranks.size();
    }

    int in_degree_signed() const {
        return asserting_cast<int>(in_degree());
    }

    size_t out_degree() const {
        return _out_ranks.size();
    }

    int out_degree_signed() const {
        return asserting_cast<int>(out_degree());
    }

    bool is_weighted() const {
        return _in_weights.has_value();
    }

    SpanType in_ranks() const {
        return _in_ranks;
    }

    SpanType out_ranks() const {
        return _out_ranks;
    }

    std::optional<SpanType> in_weights() const {
        return _in_weights;
    }

    std::optional<SpanType> out_weights() const {
        return _out_weights;
    }

    MPI_Comm create_mpi_graph_communicator(MPI_Comm comm) const {
        int const* in_weights  = is_weighted() ? _in_weights.value().data() : MPI_UNWEIGHTED;
        int const* out_weights = is_weighted() ? _out_weights.value().data() : MPI_UNWEIGHTED;

        MPI_Comm mpi_graph_comm;

        MPI_Dist_graph_create_adjacent(
            comm,
            in_degree_signed(),
            in_ranks().data(),
            in_weights,
            out_degree_signed(),
            out_ranks().data(),
            out_weights,
            MPI_INFO_NULL,
            false,
            &mpi_graph_comm
        );
        return mpi_graph_comm;
    }

private:
    SpanType                _in_ranks;
    SpanType                _out_ranks;
    std::optional<SpanType> _in_weights;
    std::optional<SpanType> _out_weights;
};

namespace internal {
template <typename EdgeRange>
constexpr bool are_edges_weighted() {
    using EdgeType = typename EdgeRange::value_type;
    return !std::is_integral_v<EdgeType>;
}
} // namespace internal

template <template <typename...> typename DefaultContainer = std::vector>
class CommunicationGraph {
public:
    CommunicationGraph() = default;

    template <typename InEdgeRange, typename OutEdgeRange>
    CommunicationGraph(InEdgeRange const& in_edges, OutEdgeRange const& out_edges) {
        constexpr bool are_in_edges_weighted  = internal::are_edges_weighted<InEdgeRange>();
        constexpr bool are_out_edges_weighted = internal::are_edges_weighted<OutEdgeRange>();
        static_assert(
            are_in_edges_weighted == are_out_edges_weighted,
            "Weight status of in and out edges is different!"
        );
        if constexpr (are_in_edges_weighted) {
            auto get_rank = [](auto const& edge) {
                auto const& [rank, _] = edge;
                return static_cast<int>(rank);
            };
            auto get_weight = [](auto const& edge) {
                auto const& [_, weight] = edge;
                return static_cast<int>(weight);
            };
            transform_elems_to_integers(in_edges, _in_ranks, get_rank);
            transform_elems_to_integers(out_edges, _out_ranks, get_rank);
            _in_weights  = DefaultContainer<int>{};
            _out_weights = DefaultContainer<int>{};
            transform_elems_to_integers(in_edges, _in_weights.value(), get_weight);
            transform_elems_to_integers(out_edges, _out_weights.value(), get_weight);
        } else {
            auto get_rank = [](auto const& edge) {
                return static_cast<int>(edge);
            };
            transform_elems_to_integers(in_edges, _in_ranks, get_rank);
            transform_elems_to_integers(out_edges, _out_ranks, get_rank);
        }
    }

    CommunicationGraph(DefaultContainer<int>&& in_ranks, DefaultContainer<int>&& out_ranks)
        : _in_ranks{std::move(in_ranks)},
          _out_ranks{std::move(out_ranks)} {}

    CommunicationGraph(
        DefaultContainer<int>&& in_ranks,
        DefaultContainer<int>&& out_ranks,
        DefaultContainer<int>&& in_weights,
        DefaultContainer<int>&& out_weights
    )
        : _in_ranks{std::move(in_ranks)},
          _out_ranks{std::move(out_ranks)},
          _in_weights{std::move(in_weights)},
          _out_weights{std::move(out_weights)} {}

    template <typename EdgeRange>
    CommunicationGraph(EdgeRange const& edges) : CommunicationGraph(edges, edges) {}

    CommunicationGraphView get_view() const {
        if (_in_weights.has_value()) {
            return CommunicationGraphView(_in_ranks, _out_ranks, _in_weights.value(), _out_weights.value());
        } else {
            return CommunicationGraphView(_in_ranks, _out_ranks);
        }
    }

    template <typename Map = std::unordered_map<size_t, size_t>>
    auto get_rank_to_out_edge_idx_mapping() {
        Map mapping;
        for (size_t i = 0; i < _in_ranks.size(); ++i) {
            size_t rank = static_cast<size_t>(_out_ranks.data()[i]);
            mapping.emplace(rank, i);
        }
        return mapping;
    }

private:
    template <typename ReadFromRange, typename WriteToContainer, typename TransformOp>
    void transform_elems_to_integers(ReadFromRange const& input, WriteToContainer& output, TransformOp&& transform_op) {
        output.resize(input.size());
        std::transform(input.data(), input.data() + static_cast<int>(input.size()), output.data(), transform_op);
    }

private:
    DefaultContainer<int>                _in_ranks;
    DefaultContainer<int>                _out_ranks;
    std::optional<DefaultContainer<int>> _in_weights;
    std::optional<DefaultContainer<int>> _out_weights;
};

/// @brief Wrapper for an MPI communicator with topology providing access to \c rank() and \c size() of the
/// communicator. The \ref Communicator is also access point to all MPI communications provided by KaMPIng.
/// @tparam DefaultContainerType The default container type to use for containers created by KaMPIng. Defaults to
/// std::vector.
/// @tparam Plugins Plugins adding functionality to KaMPIng. Plugins should be classes taking a ``Communicator``
/// template parameter and can assume that they are castable to `Communicator` from which they can
/// call any function of `kamping::Communicator`. See `test/plugin_tests.cpp` for examples.
template <
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename>
    typename... Plugins>
class DistributedGraphCommunicator
    : public TopologyCommunicator<DefaultContainerType>,
      public Plugins<DistributedGraphCommunicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    /// @brief Type of the default container type to use for containers created inside operations of this
    /// communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    template <typename Communicator>
    DistributedGraphCommunicator(Communicator const& comm, CommunicationGraphView comm_graph_view)
        : TopologyCommunicator<DefaultContainerType>(
            comm_graph_view.create_mpi_graph_communicator(comm.mpi_communicator())
        ),
          _in_degree(comm_graph_view.in_degree()),
          _out_degree(comm_graph_view.out_degree()),
          _is_weighted(comm_graph_view.is_weighted()) {}

    template <typename Communicator>
    DistributedGraphCommunicator(Communicator const& comm, CommunicationGraph<DefaultContainerType> const& comm_graph)
        : DistributedGraphCommunicator(comm, comm_graph.get_view()) {}

    auto get_communication_graph() {
        DefaultContainerType<int> in_ranks;
        in_ranks.resize(in_degree());
        DefaultContainerType<int> out_ranks;
        out_ranks.resize(out_degree());
        DefaultContainerType<int> in_weights;
        DefaultContainerType<int> out_weights;
        if (is_weighted()) {
            in_weights.resize(in_degree());
            out_weights.resize(out_degree());
        }

        MPI_Dist_graph_neighbors(
            this->_comm,
            in_degree_signed(),
            in_ranks.data(),
            is_weighted() ? in_weights.data() : MPI_UNWEIGHTED,
            out_degree_signed(),
            out_ranks.data(),
            is_weighted() ? out_weights.data() : MPI_UNWEIGHTED
        );
        if (is_weighted()) {
            return CommunicationGraph<DefaultContainerType>(
                std::move(in_ranks),
                std::move(out_ranks),
                std::move(in_weights),
                std::move(out_weights)
            );
        } else {
            return CommunicationGraph<DefaultContainerType>(std::move(in_ranks), std::move(out_ranks));
        }
    }

    bool is_weighted() const {
        return _is_weighted;
    }

    size_t in_degree() const {
        return _in_degree;
    }

    int in_degree_signed() const {
        return asserting_cast<int>(_in_degree);
    }

    size_t out_degree() const {
        return _out_degree;
    }

    int out_degree_signed() const {
        return asserting_cast<int>(_out_degree);
    }

private:
    size_t _in_degree;
    size_t _out_degree;
    bool   _is_weighted;
};
} // namespace kamping
