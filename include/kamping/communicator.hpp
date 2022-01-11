// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <mpi.h>

namespace kamping {

/// @brief Wrapper for MPI communicator providing access to \ref rank() and \ref size() of the communicator. The \ref
/// Communicator is also access point to all MPI communications provided by KaMPI.ng.
class Communicator {
public:
    /// @brief Default constructor not specifying any MPI communicator and using \c MPI_COMMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) {}

    /// @brief Constructor where an MPI communicator has to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    explicit Communicator(MPI_Comm comm) : Communicator(comm, 0) {}

    /// @brief Constructor where an MPI communicator and the default root have to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param root Default root that is used by MPI operations requiring a root.
    explicit Communicator(MPI_Comm comm, int root) : _rank(get_mpi_rank(comm)), _size(get_mpi_size(comm)), _comm(comm) {
        if (comm == MPI_COMM_NULL) {
            /// @todo Throw or assert
            std::abort();
        }
        this->root(root);
    }

    /// @brief Rank of the current MPI process in the communicator.
    /// @return Rank of the current MPI process in the communicator.
    int rank() const {
        return _rank;
    }

    /// @brief Number of MPI processes in this communicator.
    /// @return Number of MPI processes in this communicator.
    int size() const {
        return _size;
    }

    /// @brief MPI communicator corresponding to this communicator.
    /// @return MPI communicator corresponding to this communicator.
    MPI_Comm mpi_communicator() const {
        return _comm;
    }

    /// @brief Set a new root for MPI operations that require a root.
    /// @param new_root The new default root.
    void root(int const new_root) {
        /// @todo Assert or Throw if not all MPI processes in this communicator have the same root.
        if (!is_valid_rank(new_root)) {
            std::abort();
        }
        _root = new_root;
    }

    /// @brief Default root for MPI operations that require a root.
    /// @return Default root for MPI operations that require a root.
    int root() const {
        return _root;
    }

    /// @brief Split the communicator in different \colors.
    /// @param color All ranks that have the same color will be in the same new communicator.
    /// @param key By default, ranks in the new communicator are determined by the underlying MPI library (if \c key is
    /// 0). Otherwise, ranks are ordered the same way the keys are ordered.
    /// @return \ref Communicator wrapping the newly split MPI communicator.
    Communicator split(int const color, int const key = 0) const {
        MPI_Comm new_comm;
        MPI_Comm_split(_comm, color, key, &new_comm);
        return Communicator(new_comm);
    }

    /// @brief Convert a rank from this communicator to the rank in another communicator.
    /// @param rank The rank in this communicator
    /// @param otherComm The communicator to convert the rank to
    /// @return The rank in otherComm
    int convert_rank_to_communicator(int const rank, Communicator const& otherComm) const {
        MPI_Group myGroup;
        MPI_Comm_group(_comm, &myGroup);
        MPI_Group otherGroup;
        MPI_Comm_group(otherComm._comm, &otherGroup);
        int rankInOtherComm;
        MPI_Group_translate_ranks(myGroup, 1, &rank, otherGroup, &rankInOtherComm);
        return rankInOtherComm;
    }

    /// @brief Convert a rank from another communicator to the rank in this communicator.
    /// @param rank The rank in otherComm
    /// @param otherComm The communicator to convert the rank from
    /// @return The rank in this communicator
    int convert_rank_from_communicator(int const rank, Communicator const& otherComm) const {
        return otherComm.convert_rank_to_communicator(rank, *this);
    }

    /// @brief Computes a rank that is \c distance ranks away from this MPI thread's current rank and checks if this is
    /// valid rank in this communicator.
    ///
    /// The resulting rank is valid, iff it is at least zero and less than this communicator's size. The \c distance can
    /// be negative. Unlike \ref rank_shifted_cyclic(), this does not guarantee a valid rank but can indicate if the
    /// resulting rank is not valid.
    /// @param distance Amount current rank is decreased or increased by.
    /// @return Rank if rank is in [0, size of communicator) and ASSERT/EXCEPTION? otherwise.
    int rank_shifted_checked(int const distance) const {
        if (int result = _rank + distance; is_valid_rank(result)) {
            return result;
        }
        /// @todo Make use of our assert/exception functionality.
        std::abort();
    }

    /// @brief Computes a rank that is some ranks apart from this MPI thread's rank modulo the communicator's size.
    ///
    /// When we need to compute a rank that is greater (or smaller) than this communicator's rank, we can use this
    /// function. It computes the rank that is \c distance ranks appart. However, this function always returns a valid
    /// rank, as it computes the rank in a circular fashion, i.e., \f$ new\_rank=(rank + distance) \% size \f$.
    /// @param distance Distance of the new rank to the rank of this MPI thread.
    /// @return The circular rank that is \c distance ranks apart from this MPI threads rank.
    int rank_shifted_cyclic(int const distance) const {
        return (_rank + distance) % _size;
    }

    /// @brief Checks if a rank is a valid rank for this communicator, i.e., if the rank is in [0, size).
    /// @return \c true if rank in [0,size) and \c false otherwise.
    bool is_valid_rank(int const rank) const {
        return rank >= 0 && rank < _size;
    }

private:
    /// @brief Compute the rank of the current MPI process computed using \c MPI_Comm_rank.
    /// @return Rank of the current MPI process in the communicator.
    int get_mpi_rank(MPI_Comm comm) const {
        int rank;
        MPI_Comm_rank(comm, &rank);
        return rank;
    }

    /// @brief Compute the number of MPI processes in this communicator using \c MPI_Comm_size.
    /// @return Size of the communicator.
    int get_mpi_size(MPI_Comm comm) const {
        int size;
        MPI_Comm_size(comm, &size);
        return size;
    }

    int const      _rank; ///< Rank of the MPI process in this communicator.
    int const      _size; ///< Number of MPI processes in this communicator.
    MPI_Comm const _comm; ///< Corresponding MPI communicator.

    int _root; ///< Default root for MPI operations that require a root.
};             // class communicator

} // namespace kamping
