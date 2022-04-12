// This file is part of KaMPI.ng.
//
// Copyright 2021-2022 The KaMPI.ng Authors
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

#include <cstddef>
#include <cstdlib>

#include <mpi.h>

#include "error_handling.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/kassert.hpp"

namespace kamping {

/// @brief Wrapper for MPI communicator providing access to \ref rank() and \ref size() of the communicator. The \ref
/// Communicator is also access point to all MPI communications provided by KaMPI.ng.
class Communicator : public internal::Alltoall<Communicator>,
                     public internal::Scatter<Communicator>,
                     public internal::Reduce<Communicator>,
                     public internal::Gather<Communicator>,
                     public internal::Barrier<Communicator> {
public:
    /// @brief Default constructor not specifying any MPI communicator and using \c MPI_COMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) {}

    /// @brief Constructor where an MPI communicator has to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    explicit Communicator(MPI_Comm comm) : Communicator(comm, 0) {}

    /// @brief Constructor where an MPI communicator and the default root have to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param root Default root that is used by MPI operations requiring a root.
    explicit Communicator(MPI_Comm comm, int root) : _rank(get_mpi_rank(comm)), _size(get_mpi_size(comm)), _comm(comm) {
        this->root(root);
    }

    /// @brief Rank of the current MPI process in the communicator as `int`.
    /// @return Rank of the current MPI process in the communicator as `int`.
    [[nodiscard]] int rank_signed() const {
        return asserting_cast<int>(_rank);
    }

    /// @brief Rank of the current MPI process in the communicator as `size_t`.
    /// @return Rank of the current MPI process in the communicator as `size_t`.
    [[nodiscard]] size_t rank() const {
        return _rank;
    }

    /// @brief Number of MPI processes in this communicator as `int`.
    /// @return Number of MPI processes in this communicator `int`.
    [[nodiscard]] int size_signed() const {
        return asserting_cast<int>(_size);
    }

    /// @brief Number of MPI processes in this communicator as `size_t`.
    /// @return Number of MPI processes in this communicator as `size_t`.
    [[nodiscard]] size_t size() const {
        return _size;
    }

    /// @brief MPI communicator corresponding to this communicator.
    /// @return MPI communicator corresponding to this communicator.
    [[nodiscard]] MPI_Comm mpi_communicator() const {
        return _comm;
    }

    /// @brief Set a new root for MPI operations that require a root.
    /// @param new_root The new default root.
    void root(int const new_root) {
        THROWING_KASSERT(
            is_valid_rank(new_root), "invalid root rank " << new_root << " in communicator of size " << size());
        _root = asserting_cast<size_t>(new_root);
    }

    /// @brief Set a new root for MPI operations that require a root.
    /// @param new_root The new default root.
    void root(size_t const new_root) {
        THROWING_KASSERT(
            is_valid_rank(new_root), "invalid root rank " << new_root << " in communicator of size " << size());
        _root = new_root;
    }

    /// @brief Default root for MPI operations that require a root as `size_t`.
    /// @return Default root for MPI operations that require a root as `size_t`.
    [[nodiscard]] size_t root() const {
        return _root;
    }

    /// @brief Default root for MPI operations that require a root as `int`.
    /// @return Default root for MPI operations that require a root as `int`.
    [[nodiscard]] int root_signed() const {
        return asserting_cast<int>(_root);
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    /// @param root The custom root's rank.
    [[nodiscard]] bool is_root(const int root) const {
        return rank() == asserting_cast<size_t>(root);
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    /// @param root The custom root's rank.
    [[nodiscard]] bool is_root(const size_t root) const {
        return rank() == root;
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    [[nodiscard]] bool is_root() const {
        return is_root(root());
    }

    /// @brief Split the communicator in different colors.
    /// @param color All ranks that have the same color will be in the same new communicator.
    /// @param key By default, ranks in the new communicator are determined by the underlying MPI library (if \c key is
    /// 0). Otherwise, ranks are ordered the same way the keys are ordered.
    /// @return \ref Communicator wrapping the newly split MPI communicator.
    [[nodiscard]] Communicator split(int const color, int const key = 0) const {
        MPI_Comm new_comm;
        MPI_Comm_split(_comm, color, key, &new_comm);
        return Communicator(new_comm);
    }

    /// @brief Convert a rank from this communicator to the rank in another communicator.
    /// @param rank The rank in this communicator
    /// @param other_comm The communicator to convert the rank to
    /// @return The rank in other_comm
    [[nodiscard]] int convert_rank_to_communicator(int const rank, Communicator const& other_comm) const {
        MPI_Group my_group;
        MPI_Comm_group(_comm, &my_group);
        MPI_Group other_group;
        MPI_Comm_group(other_comm._comm, &other_group);
        int rank_in_other_comm;
        MPI_Group_translate_ranks(my_group, 1, &rank, other_group, &rank_in_other_comm);
        return rank_in_other_comm;
    }

    /// @brief Convert a rank from another communicator to the rank in this communicator.
    /// @param rank The rank in other_comm
    /// @param other_comm The communicator to convert the rank from
    /// @return The rank in this communicator
    [[nodiscard]] int convert_rank_from_communicator(int const rank, Communicator const& other_comm) const {
        return other_comm.convert_rank_to_communicator(rank, *this);
    }

    /// @brief Computes a rank that is \c distance ranks away from this MPI thread's current rank and checks if this is
    /// valid rank in this communicator.
    ///
    /// The resulting rank is valid, iff it is at least zero and less than this communicator's size. The \c distance can
    /// be negative. Unlike \ref rank_shifted_cyclic(), this does not guarantee a valid rank but can indicate if the
    /// resulting rank is not valid.
    /// @param distance Amount current rank is decreased or increased by.
    /// @return Rank if rank is in [0, size of communicator) and ASSERT/EXCEPTION? otherwise.
    [[nodiscard]] size_t rank_shifted_checked(int const distance) const {
        int const result = rank_signed() + distance;
        THROWING_KASSERT(is_valid_rank(result), "invalid shifted rank " << result);
        return asserting_cast<size_t>(result);
    }

    /// @brief Computes a rank that is some ranks apart from this MPI thread's rank modulo the communicator's size.
    ///
    /// When we need to compute a rank that is greater (or smaller) than this communicator's rank, we can use this
    /// function. It computes the rank that is \c distance ranks appart. However, this function always returns a valid
    /// rank, as it computes the rank in a circular fashion, i.e., \f$ new\_rank=(rank + distance) \% size \f$.
    /// @param distance Distance of the new rank to the rank of this MPI thread.
    /// @return The circular rank that is \c distance ranks apart from this MPI threads rank.
    [[nodiscard]] size_t rank_shifted_cyclic(int const distance) const {
        int const capped_distance = distance % size_signed();
        return asserting_cast<size_t>((rank_signed() + capped_distance + size_signed()) % size_signed());
    }

    /// @brief Checks if a rank is a valid rank for this communicator, i.e., if the rank is in [0, size).
    /// @return \c true if rank in [0,size) and \c false otherwise.
    [[nodiscard]] bool is_valid_rank(int const rank) const {
        return rank >= 0 && rank < size_signed();
    }

    /// @brief Checks if a rank is a valid rank for this communicator, i.e., if the rank is in [0, size).
    /// @return \c true if rank in [0,size) and \c false otherwise.
    [[nodiscard]] bool is_valid_rank(size_t const rank) const {
        return rank < size();
    }

private:
    /// @brief Compute the rank of the current MPI process computed using \c MPI_Comm_rank.
    /// @return Rank of the current MPI process in the communicator.
    size_t get_mpi_rank(MPI_Comm comm) const {
        THROWING_KASSERT(comm != MPI_COMM_NULL, "communicator must be initialized with a valid MPI communicator");

        int rank;
        MPI_Comm_rank(comm, &rank);
        return asserting_cast<size_t>(rank);
    }

    /// @brief Compute the number of MPI processes in this communicator using \c MPI_Comm_size.
    /// @return Size of the communicator.
    size_t get_mpi_size(MPI_Comm comm) const {
        THROWING_KASSERT(comm != MPI_COMM_NULL, "communicator must be initialized with a valid MPI communicator");

        int size;
        MPI_Comm_size(comm, &size);
        return asserting_cast<size_t>(size);
    }

    size_t   _rank; ///< Rank of the MPI process in this communicator.
    size_t   _size; ///< Number of MPI processes in this communicator.
    MPI_Comm _comm; ///< Corresponding MPI communicator.

    size_t _root; ///< Default root for MPI operations that require a root.
};                // class communicator

} // namespace kamping
