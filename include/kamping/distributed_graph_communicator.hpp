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

/// @brief View on a a distributed communication graph. Each vertex is a rank and the edge define possible communication
/// links between the vertices. This view of a distributed communication graph contains views on the (potentially
/// weighted) in- and outgoing edges which are basically a sequence of neighboring ranks. Note that MPI allow this to be
/// a multi-graph.
class CommunicationGraphView {
public:
    using SpanType = kamping::Span<int const>; ///< type to be used for views on ranks and weights

    /// @brief Construct a view of an unweighted communication graph.
    ///
    /// @tparam ContiguousRange Type of the input range for in and out ranks.
    /// @param in_ranks Neighboring in ranks, i.e., ranks i for which there is an edge (i,own_rank).
    /// @param out_ranks Neighboring out ranks, i.e., ranks i for which there is an edge (own_rank, i).
    template <typename ContiguousRange>
    CommunicationGraphView(ContiguousRange const& in_ranks, ContiguousRange const& out_ranks)
        : _in_ranks{in_ranks.data(), in_ranks.size()},
          _out_ranks{out_ranks.data(), out_ranks.size()} {}
    /// @brief Construct a view of an unweighted communication graph.
    ///
    /// @tparam ContiguousRange Type of the input range for in and out ranks.
    /// @param in_ranks Neighboring in ranks, i.e., ranks i for which there is an edge (i,own_rank).
    /// @param out_ranks Neighboring out ranks, i.e., ranks i for which there is an edge (own_rank, i).
    /// @param in_weights Weights associcated to neighboring in ranks.
    /// @param out_weights Weights associcated to neighboring out ranks.
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

    /// @brief Returns in degree of the rank, i.e. the number of in-going edges/communication links towards the rank.
    /// @return In degree of the rank.
    size_t in_degree() const {
        return _in_ranks.size();
    }

    /// @brief Returns in degree of the rank, i.e. the number of in-going edges/communication links towards the rank.
    /// @return Signed in degree of the rank.
    int in_degree_signed() const {
        return asserting_cast<int>(in_degree());
    }

    /// @brief Returns out degree of the rank, i.e. the number of out-going edges/communication links starting at the
    /// rank.
    /// @return Out degree of the rank.
    size_t out_degree() const {
        return _out_ranks.size();
    }

    /// @brief Returns out degree of the rank, i.e. the number of out-going edges/communication links starting at the
    /// rank.
    /// @return Signed out degree of the rank.
    int out_degree_signed() const {
        return asserting_cast<int>(out_degree());
    }

    /// @brief Returns whether the communication graph is weighted or not.
    bool is_weighted() const {
        return _in_weights.has_value();
    }

    /// @brief Returns view on the in-going edges.
    SpanType in_ranks() const {
        return _in_ranks;
    }

    /// @brief Returns view on the out-going edges.
    SpanType out_ranks() const {
        return _out_ranks;
    }

    /// @brief Returns view on the in-going edge weights if present.
    std::optional<SpanType> in_weights() const {
        return _in_weights;
    }

    /// @brief Returns view on the out-going edge weights if present.
    std::optional<SpanType> out_weights() const {
        return _out_weights;
    }

    /// @brief Creates a distributed graph communicator based on the view of the given communication graph using \c
    /// MPI_Dist_graph_create_adjacent.
    ///
    /// @param comm MPI comm on which the graph topology will be applied.
    /// @return MPi comm with associated graph topology.
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
    SpanType                _in_ranks;    ///< View on in-going edges.
    SpanType                _out_ranks;   ///< View on out-going edges.
    std::optional<SpanType> _in_weights;  ///< View on in-going edge weights.
    std::optional<SpanType> _out_weights; ///< View on out-going edge weights.
};

namespace internal {
/// @brief Returns whether a given range of edges is weighted or not at compile time
/// @tparam EdgeRange Range type to be checked.
template <typename EdgeRange>
constexpr bool are_edges_weighted() {
    using EdgeType = typename EdgeRange::value_type;
    return !std::is_integral_v<EdgeType>;
}
} // namespace internal

/// @brief A Distributed communication graph. Each vertex of the graph corresponds to a rank and each edge (i,j) connect
/// two ranks i and j which can communicate with each other. Note that MPI allow multiple edges between the same ranks i
/// and j, i.e. the distributed communication graph can be a multi-graph. Each rank holds its local view on the
/// communication graph, i.e., it knows its neighboring vertices/ranks.
template <template <typename...> typename DefaultContainer = std::vector>
class CommunicationGraph {
public:
    /// @brief Default constructor.
    CommunicationGraph() = default;

    /// @brief Constructs a communication graph based on a range of in-going and out-going edges which might be
    /// weighted. A unweighted edge is simply an integer, wherea a weighted edge is a pair-like object consisting of two
    /// integers which can be decomposed by a structured binding via `auto [rank, weight] = weighted_edge;`.
    ///
    /// @tparam InEdgeRange Range type of in-going edges.
    /// @tparam InEdgeRange Range type of out-going edges.
    /// @param in_edges Range of in-going edges.
    /// @param out_edges Range of out-going edges.
    ///
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

    /// @brief Constructs a communication graph based on a range of unweighted in-going and out-going edges.
    CommunicationGraph(DefaultContainer<int>&& in_ranks, DefaultContainer<int>&& out_ranks)
        : _in_ranks{std::move(in_ranks)},
          _out_ranks{std::move(out_ranks)} {}

    /// @brief Constructs a communication graph based on a range of weighted in-going and out-going edges.
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

    /// @brief Constructs a communication graph based on a range of potentially weighted symmetric edges.
    template <typename EdgeRange>
    CommunicationGraph(EdgeRange const& edges) : CommunicationGraph(edges, edges) {}

    /// @brief Returns a view of the communication graph.
    CommunicationGraphView get_view() const {
        if (_in_weights.has_value()) {
            return CommunicationGraphView(_in_ranks, _out_ranks, _in_weights.value(), _out_weights.value());
        } else {
            return CommunicationGraphView(_in_ranks, _out_ranks);
        }
    }

    /// @brief For each rank n for which there is (at least one) outgoing edge (own_rank, r), returns a mapping
    /// r -> index of an edge (own_rank, r). This may be useful for collective communication primitives like
    /// `MPI_Neighbor_alltoall` when one wants to calculate the position within the send_buffer to send data to a rank
    /// r.
    ///
    /// @return Returns the mapping
    template <typename Map = std::unordered_map<size_t, size_t>>
    auto get_rank_to_out_edge_idx_mapping() {
        Map mapping;
        for (size_t i = 0; i < _out_ranks.size(); ++i) {
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
    DefaultContainer<int>                _in_ranks;    ///< In-going edges.
    DefaultContainer<int>                _out_ranks;   ///< Out-going edges.
    std::optional<DefaultContainer<int>> _in_weights;  ///< Weights of in-going edges if present.
    std::optional<DefaultContainer<int>> _out_weights; ///< Weights of out-going edges if present.
};

/// @brief Wrapper for an MPI communicator with a (distributed) graph topology providing access to \c rank() and \c
/// size() of the communicator. The \ref Communicator is also access point to all MPI communications provided by
/// KaMPIng.
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

    /// @brief Construtor based on a given communicator and a view of a communication graph.
    ///
    /// @tparam Communicator Type of communicator.
    /// @param comm Communicator for which a graph topology shall be added.
    /// @param comm_graph_view View on the communication graph which will be added to the given communicator.
    template <typename Communicator>
    DistributedGraphCommunicator(Communicator const& comm, CommunicationGraphView comm_graph_view)
        : TopologyCommunicator<DefaultContainerType>(
              comm_graph_view.in_degree(),
              comm_graph_view.out_degree(),
              comm_graph_view.create_mpi_graph_communicator(comm.mpi_communicator())
          ),
          _is_weighted(comm_graph_view.is_weighted()) {}

    /// @brief Construtor based on a given communicator and a communication graph.
    ///
    /// @tparam Communicator Type of communicator.
    /// @param comm Communicator for which a graph topology shall be added.
    /// @param comm_graph Communication graph which will be added to the given communicator.
    template <typename Communicator>
    DistributedGraphCommunicator(Communicator const& comm, CommunicationGraph<DefaultContainerType> const& comm_graph)
        : DistributedGraphCommunicator(comm, comm_graph.get_view()) {}

    /// @brief Returns the communicators underlying communication graph by calling `MPI_Dist_graph_neighbors`.
    auto get_communication_graph() {
        DefaultContainerType<int> in_ranks;
        in_ranks.resize(this->in_degree());
        DefaultContainerType<int> out_ranks;
        out_ranks.resize(this->out_degree());
        DefaultContainerType<int> in_weights;
        DefaultContainerType<int> out_weights;
        if (is_weighted()) {
            in_weights.resize(this->in_degree());
            out_weights.resize(this->out_degree());
        }

        MPI_Dist_graph_neighbors(
            this->_comm,
            this->in_degree_signed(),
            in_ranks.data(),
            is_weighted() ? in_weights.data() : MPI_UNWEIGHTED,
            this->out_degree_signed(),
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

    /// @brief Returns whether the communicator's underlying communication graph is weighted.
    /// @return True if weighted, false otherwise.
    bool is_weighted() const {
        return _is_weighted;
    }

private:
    size_t _in_degree;   ///< In degree of the rank within the communication graph.
    size_t _out_degree;  ///< Out degree of the rank within the communication graph.
    bool   _is_weighted; ///< Weight status of the communication graph.
};
} // namespace kamping
