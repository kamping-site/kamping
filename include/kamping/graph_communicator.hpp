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
#include <kamping/topology_communicator.hpp>

namespace kamping {
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
class GraphCommunicator : public TopologyCommunicator<DefaultContainerType>,
                          public Plugins<GraphCommunicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    /// @brief Type of the default container type to use for containers created inside operations of this communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    template <typename Communicator, typename EdgeContainer>
    GraphCommunicator(Communicator const& comm, EdgeContainer const& in_edges, EdgeContainer const& out_edges)
        : TopologyCommunicator<DefaultContainerType>(creation_helper(comm, in_edges, out_edges)),
          _in_degree(in_edges.size()),
          _out_degree(out_edges.size()) {
        auto get_rank = [&](auto const& edge) {
            using edge_value_type        = typename EdgeContainer::value_type;
            constexpr bool is_unweighted = std::is_same_v<edge_value_type, int>;
            if constexpr (is_unweighted) {
                return edge;
            } else {
                const auto& [rank, weight] = edge;
                return rank;
            }
        };

        for (size_t i = 0; i < in_edges.size(); ++i) {
            auto const& edge = in_edges.data()[i];
            _graph_rank_to_rank.emplace_back(get_rank(edge));
        }

        for (size_t i = 0; i < out_edges.size(); ++i) {
            auto const& edge = in_edges.data()[i];
            _rank_to_graph_rank.emplace(get_rank(edge), i);
        }
    }

    template <typename Communicator, typename Edges>
    GraphCommunicator(Communicator const& comm, Edges const& edges) : GraphCommunicator(comm, edges, edges) {}

    size_t in_degree() const {
        return _in_degree;
    }

    size_t in_degree_signed() const {
        return asserting_cast<int>(_in_degree);
    }

    size_t out_degree() const {
        return _out_degree;
    }

    size_t out_degree_signed() const {
        return asserting_cast<int>(_out_degree);
    }

    size_t out_graph_rank(size_t global_rank) const {
        auto it = _rank_to_graph_rank.find(global_rank);
        KASSERT(it != _rank_to_graph_rank.end(), "global rank has not been defined as an (out) communication partner");
        return it->second;
    }

    int out_graph_rank_signed(size_t global_rank) const {
        return asserting_cast<int>(out_graph_rank(global_rank));
    }

    size_t in_global_rank(size_t graph_rank) const {
        KASSERT(
            graph_rank < _graph_rank_to_rank.size(),
            "Requested graph rank index is greater than (input) communication graph."
        );
        return _graph_rank_to_rank[graph_rank];
    }

private:
    template <typename EdgeContainer>
    class EdgeConverter {
    public:
        using edge_value_type               = typename EdgeContainer::value_type;
        static constexpr bool is_unweighted = std::is_same_v<edge_value_type, int>;

        EdgeConverter(EdgeContainer const& edges) : _size{edges.size()} {
            if constexpr (is_unweighted) {
                _ranks   = edges.data();
                _weights = MPI_UNWEIGHTED;
            } else {
                _ranks.resize(edges.size());
                _weights.resize(edges.size());
                for (size_t i = 0; i < edges.size(); ++i) {
                    auto const [rank, weight] = edges.data()[i];
                    _ranks.data()[i]          = rank;
                    _weights()[i]             = weight;
                }
            }
        }

        int const* get_ranks_ptr() const {
            if constexpr (is_unweighted) {
                return _ranks;
            } else {
                return _ranks.data();
            }
        }

        int* get_weights_ptr() const {
            if constexpr (is_unweighted) {
                return MPI_UNWEIGHTED;
            } else {
                return _weights.data();
            }
        }

        int size_signed() const {
            return asserting_cast<int>(_size);
        }

    private:
        size_t                                                                   _size;
        std::conditional_t<is_unweighted, int const*, DefaultContainerType<int>> _ranks;
        std::conditional_t<is_unweighted, int*, DefaultContainerType<int>>       _weights;
    };

    template <typename Communicator, typename Edges>
    static MPI_Comm creation_helper(Communicator const& comm, Edges const& in_edges, Edges const& out_edges) {
        MPI_Comm graph_mpi_comm;

        EdgeConverter<Edges> const converted_in_edges(in_edges);
        EdgeConverter<Edges> const converted_out_edges(out_edges);

        MPI_Dist_graph_create_adjacent(
            comm.mpi_communicator(),
            converted_in_edges.size_signed(),
            converted_in_edges.get_ranks_ptr(),
            converted_in_edges.get_weights_ptr(),
            converted_out_edges.size_signed(),
            converted_out_edges.get_ranks_ptr(),
            converted_out_edges.get_weights_ptr(),
            MPI_INFO_NULL,
            false,
            &graph_mpi_comm
        );
        return graph_mpi_comm;
    }

private:
    size_t                             _in_degree;
    size_t                             _out_degree;
    std::vector<size_t>                _graph_rank_to_rank;
    std::unordered_map<size_t, size_t> _rank_to_graph_rank;
};
} // namespace kamping
