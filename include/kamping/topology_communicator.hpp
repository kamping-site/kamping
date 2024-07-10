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

#include <kamping/collectives/collectives_helpers.hpp>
#include <kamping/communicator.hpp>

namespace kamping {
/// @brief  A \ref Communicator which possesses an additional virtual topology and supports neighborhood collectives (on
/// the topology). A virtual topology can be defined by a communication graph: each MPI rank corresponds to a vertex in
/// the graph and an edge (i,j) defines a (directed) communication link from rank i to rank j. Such a topolgy can be
/// used to model frequent (sparse) communication patterns and there are specialized (neighborhood) collective
/// operations, e.g., MPI_Neighbor_alltoall etc., exploiting this structure.
///
/// @tparam DefaultContainerType The default container type to use for containers created by KaMPIng. Defaults to
/// std::vector.
/// @tparam Plugins Plugins adding functionality to KaMPIng. Plugins should be classes taking a ``Communicator``
/// template parameter and can assume that they are castable to `Communicator` from which they can
/// call any function of `kamping::Communicator`. See `test/plugin_tests.cpp` for examples.
template <
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename>
    typename... Plugins>
class TopologyCommunicator
    : public Communicator<DefaultContainerType>,
      public Plugins<TopologyCommunicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    /// @brief Type of the default container type to use for containers created inside operations of this communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    /// @brief Returns the in degree of the process' rank, i.e. the number of in-going edges/communication links towards
    /// the rank.
    /// @return Number of in-going edges as unsigned integer.
    size_t in_degree() const {
        return _in_degree;
    }
    /// @brief Returns the in degree of the process' rank, i.e. the number of in-going edges/communication links towards
    /// the rank.
    /// @return Number of in-going edges as signed integer.
    int in_degree_signed() const {
        return asserting_cast<int>(_in_degree);
    }

    /// @brief Returns the out degree of the process' rank, i.e. the number of out-going edges/communication links
    /// starting at the rank.
    /// @return Number of out-going edges as unsigned integer.
    size_t out_degree() const {
        return _out_degree;
    }

    /// @brief Returns the out degree of the process' rank, i.e. the number of out-going edges/communication links
    /// starting at the rank.
    /// @return Number of out-going edges as signed integer.
    int out_degree_signed() const {
        return asserting_cast<int>(_out_degree);
    }

    template <typename... Args>
    auto neighbor_alltoall(Args... args) const;

protected:
    using Communicator<DefaultContainerType>::Communicator;

    /// @brief topolgy constructor using \c MPI_COMM_WORLD by default.
    ///
    /// @param in_degree In degree of the process' rank in the underlying communication graph.
    /// @param out_degree Out degree of the process' rank in the underlying communication graph.
    TopologyCommunicator(size_t in_degree, size_t out_degree)
        : TopologyCommunicator(in_degree, out_degree, MPI_COMM_WORLD) {}

    /// @brief topolgy constructor where an MPI communicator has to be specified.
    ///
    /// @param in_degree In degree of the process' rank in the underlying communication graph.
    /// @param out_degree Out degree of the process' rank in the underlying communication graph.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param take_ownership Whether the Communicator should take ownership of comm, i.e. free it in the destructor.
    explicit TopologyCommunicator(size_t in_degree, size_t out_degree, MPI_Comm comm, bool take_ownership = false)
        : TopologyCommunicator<DefaultContainerType>(in_degree, out_degree, comm, 0, take_ownership) {}

    /// @brief topolgy constructor where an MPI communicator and the default root rank have to be specified.
    ///
    /// @param in_degree In degree of the process' rank in the underlying communication graph.
    /// @param out_degree Out degree of the process' rank in the underlying communication graph.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param root Default root that is used by MPI operations requiring a root.
    /// @param take_ownership Whether the Communicator should take ownership of comm, i.e. free it in the destructor.
    explicit TopologyCommunicator(
        size_t in_degree, size_t out_degree, MPI_Comm comm, int root, bool take_ownership = false
    )
        : Communicator<DefaultContainerType>(comm, root, take_ownership),
          _in_degree{in_degree},
          _out_degree{out_degree} {}

private:
    size_t _in_degree;  ///< In degree of the underlying communication graph.
    size_t _out_degree; ///< Out degree of the underlying communication graph.
};
} // namespace kamping
