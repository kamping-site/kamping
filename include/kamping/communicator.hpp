#pragma once

#include <mpi.h>

namespace kamping {

  /// @brief Wrapper for MPI communicator providing access to \ref rank() and \ref size() of the communicator.
  class Communicator {

  public:
    /// @brief Default constructor not specifying any MPI communicator and using \c MPI_COMMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) { }

    /// @brief Constructor where an MPI communicator has to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    explicit Communicator(MPI_Comm comm) : _rank(get_mpi_rank(comm)),
                                           _size(get_mpi_size(comm)),
                                           _comm(comm) { }

    /// @brief Rank of the current MPI process in the communicator.
    /// @return Rank of the current MPI process in the communicator.
    int rank() const {
      return _rank;
    }

    /// @brief Size of the communicator.
    /// @return Size of the communicator.
    int size() const {
      return _size;
    }

    /// @brief MPI communicator corresponding to this communicator.
    /// @return MPI communicator corresponding to this communicator.
    MPI_Comm mpi_communicator() const {
      return _comm;
    }

  private:

    /// @brief Compute the rank of the current MPI process computed using \c MPI_Comm_rank.
    /// @return Rank of the current MPI process in the communicator.
    int get_mpi_rank(MPI_Comm comm) const {
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }

    /// @brief Compute the size of the communicator using \c MPI_Comm_size.
    /// @return Size of the communicator.
    int get_mpi_size(MPI_Comm comm) const {
      int size;
      MPI_Comm_size(comm, &size);
      return size;
    }

    int const _rank; ///< Rank of the MPI process in this communicator.
    int const _size; ///< Size of this communicator.
    MPI_Comm const _comm; ///< Corresponding MPI communicator.
  }; // class communicator

} // namespace kamping
