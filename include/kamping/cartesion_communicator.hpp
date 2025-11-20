#pragma once

#include <kamping/communicator.hpp>
#include <kamping/topology_communicator.hpp>
#include <mpi.h>

namespace kamping {

template <
    std::size_t N,
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename> typename... Plugins>
class CartesianCommunicator
    : public TopologyCommunicator<DefaultContainerType>,
      public Plugins<CartesianCommunicator<N, DefaultContainerType, Plugins...>, DefaultContainerType>... {
    /// @brief Type of the default container type to use for containers created inside operations of this
    /// communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    template <typename Communicator>
    CartesianCommunicator(
        Communicator const& comm, bool periodic = false, bool reorder = false, bool take_ownership = false
    )
        : TopologyCommunicator<DefaultContainerType>(N, N, [&] {
              MPI_Comm           comm_cart = MPI_COMM_NULL;
              std::array<int, N> dims;
              std::array<int, N> periodic;
              std::fill(periodic.begin(), periodic.end(), periodic);
              MPI_Dims_create(comm.size(), N, dims.data());
              MPI_Cart_create(comm.mpi_communicator(), N, dims.data(), periodic.data(), reorder, &comm_cart);
              return comm_cart;
          }()) {}

    auto coords(int rank) {
        std::array<int, N> coords;
        MPI_Cart_coords(this->mpi_communicator(), rank, N, coords.data());
        // TODO unpack array to tuple
    }

    int rank(std::array<int, N> const& coords) {
        int rank;
        MPI_Cart_rank(this->mpi_communicator(), coords.data(), &rank);
        return rank;
    }

    
    int rank(std::size_t indices...) {
        std::array<int, N> coords{static_cast<int>(indices)...};
        return rank(coords);
    }
};

} // namespace kamping
