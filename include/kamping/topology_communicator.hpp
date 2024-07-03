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
class TopologyCommunicator
    : public Communicator<DefaultContainerType>,
      public Plugins<TopologyCommunicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    using Communicator<DefaultContainerType>::Communicator;
    TopologyCommunicator(size_t in_degree, size_t out_degree)
        : TopologyCommunicator(in_degree, out_degree, MPI_COMM_WORLD) {}

    explicit TopologyCommunicator(size_t in_degree, size_t out_degree, MPI_Comm comm, bool take_ownership = false)
        : TopologyCommunicator<DefaultContainerType>(in_degree, out_degree, comm, 0, take_ownership) {}

    explicit TopologyCommunicator(
        size_t in_degree, size_t out_degree, MPI_Comm comm, int root, bool take_ownership = false
    )
        : Communicator<DefaultContainerType>(comm, root, take_ownership),
          _in_degree{in_degree},
          _out_degree{out_degree} {}

    /// @brief Type of the default container type to use for containers created inside operations of this communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

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

    template <typename... Args>
    auto neighbor_alltoall(Args... args) const;

private:
    size_t _in_degree;
    size_t _out_degree;
};
} // namespace kamping
