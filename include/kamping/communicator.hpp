#pragma once

#include <mpi.h>

namespace kamping {

  /// \brief Wrapper for MPI communicator providing access to \ref rank() and \ref size() of the communicator.
  class Communicator {

  public:
    /// \brief Default constructor not specifying any MPI communicator and using \c MPI_COMMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) { }

    /// \brief Constructor where an MPI communicator has to be specified
    explicit Communicator(MPI_Comm comm) : _comm(comm) {
      MPI_Comm_rank(_comm, &_rank);
      MPI_Comm_size(_comm, &_size);
    }

    /// \brief Rank of the current MPI process in the communicator.
    /// \return Rank of the current MPI process in the communicator.
    int rank() const {
      return _rank;
    }

    /// \brief Size of the communicator.
    /// \return Size of the communicator.
    int size() const {
      return _size;
    }

    /// \brief MPI communicator corresponding to this communicator.
    /// \return MPI communicator corresponding to this communicator.
    MPI_Comm mpi_communicator() const {
      return _comm;
    }

  private:
    int _rank; ///< Rank of the MPI process in this communicator.
    int _size; ///< Size of this communicator.
    MPI_Comm _comm; ///< Corresponding MPI communicator.
  }; // class communicator

} // namespace kamping
