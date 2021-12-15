#pragma once

#include <mpi.h>

namespace kamping {

/// @brief Wrapper for MPI communicator providing access to \ref rank() and \ref size() of the communicator.
class Communicator {
public:
    /// @brief Default constructor not specifying any MPI communicator and using \c MPI_COMMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) {}

    /// @brief Constructor where an MPI communicator has to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    explicit Communicator(MPI_Comm comm) : _rank(get_mpi_rank(comm)), _size(get_mpi_size(comm)), _comm(comm) {}

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

    /// @brief Increases the current rank by \c distance and checks if the resulting rank is valid in this communicator.
    ///
    /// The resulting rank is valid, iff it is at least zero and less than this communicator's size. The \c distance can
    /// be negative. Unlike \ref rank_advance_cyclic(), this does not guarantee a valid rank but can indicate if the
    /// resulting rank is not valid.
    /// @param distance Amount current rank is decreased or increased by.
    /// @return Rank if rank is in [0, size of communicator) and ASSERT/EXCEPTION? otherwise.
    int rank_advance_bound_checked(int const distance) const {
        if (int result = _rank + distance; result < _size && result >= 0) {
            return result;
        }
        /// @todo Make use of our assert/exception functionality.
        std::abort();
    }

    /// @brief Cyclically compute rank with distance \c distance.
    ///
    /// If there are not \c distance more (or less) ranks, the computation considers the ranks cyclically. Note that
    /// unlike \ref rank_advance_bound_checked() this always results in a valid rank.
    /// @param distance Amount current rank is decreased or increased by.
    /// @return The cyclic rank \c distance ranks apart from the current rank.
    int rank_advance_cyclic(int const distance) const {
        return (_rank + distance) % _size;
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

    int const      _rank; ///< Rank of the MPI process in this communicator.
    int const      _size; ///< Size of this communicator.
    MPI_Comm const _comm; ///< Corresponding MPI communicator.
};                        // class communicator

} // namespace kamping
