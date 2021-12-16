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

/// @file
/// @brief File containing the main KaMPI.ng functionality
#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/template_magic_helpers.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

struct Rank {
    int rank;
};
namespace internal {
///@brief Use this type if one of the template parameters of MPIResult is not used for a specific wrapped \c MPI call.
struct BufferCategoryNotUsed {};
} // namespace internal

///@brief MPIResult contains the result of a \c MPI call wrapped by KaMPI.ng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transfered to KaMPI.ng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every wrapped \c MPI call.
/// If a specific call does not use a buffer category, you have to provide internal::BufferCategoryNotUsed instead.
///
///@tparam RecBuf Buffer type containing the received elements.
///@tparam RecCounts Buffer type containing the numbers of received elements.
///@tparam RecDispls Buffer type containing the displacements of the received elements.
///@tparam SendDispls Buffer type containing the displacements of the sent elements.
///@tparam MPIStatusObject Buffer type containing the \c MPI status object(s).
template <class RecvBuf, class RecvCounts, class RecvDispls, class SendDispls, class MPIStatusObject>
class MPIResult {
public:
    MPIResult(
        RecvBuf&& recv_buf, RecvCounts&& recv_counts, RecvDispls&& recv_displs, SendDispls&& send_displs,
        MPIStatusObject&& mpi_status)
        : _recv_buffer(std::forward<recv_buf>(recv_buf)),
          _recv_counts(std::forward<RecvCounts>(recv_counts)),
          _recv_displs(std::forward<RecvDispls>(recv_displs)),
          _send_displs(std::forward<SendDispls>(send_displs)),
          _mpi_status(std::forward<MPIStatusObject>(mpi_status)) {}

    template <typename U = RecvBuf, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_buffer() {
        return _recv_buffer.extract();
    }

    template <typename U = RecvCounts, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_counts() {
        return _recv_counts.extract();
    }

    template <typename U = RecvDispls, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_displs() {
        return _recv_displs.extract();
    }

    template <typename U = SendDispls, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_send_displs() {
        return _send_displs.extract();
    }

    template <typename U = MPIStatusObject, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_mpi_status() {
        return _mpi_status.extract();
    }

private:
    RecvBuf         _recv_buffer;
    RecvCounts      _recv_counts;
    RecvDispls      _recv_displs;
    SendDispls      _send_displs;
    MPIStatusObject _mpi_status;
};
class MPIContext {
public:
    // enum class SendMode { normal, buffered, synchronous };
    // explicit MPIContext(MPI_Comm comm) : comm_{comm} {
    //     MPI_Comm_rank(comm_, &rank_);
    //     MPI_Comm_size(comm_, &size_);
    // }

    // template <class recvBuffType, class recvCountsType, class recvDisplsType>
    // struct gatherv_output {
    //     gatherv_output(recvBuffType&& recvBuff, recvCountsType&& recvCounts,
    //     recvDisplsType&& recvDispls)
    //         : _recvBuff(std::forward<recvBuffType>(recvBuff)),
    //           _recvCounts(std::forward<recvCountsType>(recvCounts)),
    //           _recvDispls(std::forward<recvDisplsType>(recvDispls)) {}

    //    // TODO Try to do this by checking whether extract() exists
    //    template <typename recvBuffType2 = recvBuffType,
    //    std::enable_if_t<recvBuffType2::isExtractable, bool> = true>
    //    decltype(auto) extractRecvBuff() {
    //        return _recvBuff.extract();
    //    }

    //    template <
    //        typename recvCountsType2 = recvCountsType,
    //        std::enable_if_t<recvCountsType2::isExtractable, bool> = true>
    //    decltype(auto) extractRecvCounts() {
    //        return _recvCounts.extract();
    //    }

    //    template <
    //        typename recvDisplsType2 = recvDisplsType,
    //        std::enable_if_t<recvDisplsType2::isExtractable, bool> = true>
    //    decltype(auto) extractRecvDispls() {
    //        return _recvDispls.extract();
    //    }

    // private:
    //     recvBuffType   _recvBuff;
    //     recvCountsType _recvCounts;
    //     recvDisplsType _recvDispls;
    // };


    // template <class... Args>
    // auto gatherv(Args&&... args) {
    //     auto sendBuf    =
    //     select_trait<ptraits::in>(std::forward<Args>(args)...); using send_type
    //     = typename decltype(sendBuf)::value_type;

    //    // Get receive buffer and use new vector if none is given
    //    auto recvBuf    =
    //    select_trait<ptraits::out>(std::forward<Args>(args)...,
    //    out(new_vector<send_type>())); using recv_type = typename
    //    decltype(recvBuf)::value_type;

    //    auto recvCountsContainer =
    //        select_trait<ptraits::recvCounts>(std::forward<Args>(args)...,
    //        recv_counts(new_vector<int>()));
    //    static_assert(
    //        std::is_same<typename decltype(recvCountsContainer)::value_type,
    //        int>::value, "Recv counts must be int");

    //    auto recvDisplsContainer =
    //        select_trait<ptraits::recvDispls>(std::forward<Args>(args)...,
    //        recv_displs(new_vector<int>()));
    //    static_assert(
    //        std::is_same<typename decltype(recvDisplsContainer)::value_type,
    //        int>::value, "Recv displacements must be int");

    //    // Select root. Defaults to 0
    //    // TODO let user choose default root for context
    //    auto rootPE = select_trait<ptraits::root>(std::forward<Args>(args)...,
    //    root(0));

    //    //  Gather send counts at root
    //    // TODO don't do this if the user supplies send counts at root
    //    int mySendCount = static_cast<int>(sendBuf.get().size);

    //    int*       recvCountsPtr;
    //    int*       recvDisplsPtr;
    //    recv_type* recvPtr = nullptr;
    //    if (rank_ == rootPE.getRoot()) {
    //        recvCountsPtr =
    //        recvCountsContainer.get_ptr(static_cast<std::size_t>(size_));
    //        recvDisplsPtr =
    //        recvDisplsContainer.get_ptr(static_cast<std::size_t>(size_));
    //        MPI_Gather(&mySendCount, 1, MPI_INT, recvCountsPtr, 1, MPI_INT,
    //        rootPE.getRoot(), comm_);

    //        std::exclusive_scan(recvCountsPtr, recvCountsPtr + size_,
    //        recvDisplsPtr, 0);

    //        int recvSize = *(recvDisplsPtr + size_ - 1) + *(recvCountsPtr +
    //        size_ - 1); recvPtr      =
    //        recvBuf.get_ptr(static_cast<std::size_t>(recvSize));
    //    } else {
    //        recvCountsPtr = recvCountsContainer.get_ptr(0);
    //        recvDisplsPtr = recvDisplsContainer.get_ptr(0);
    //        recvPtr       = recvBuf.get_ptr(0);
    //        MPI_Gather(&mySendCount, 1, MPI_INT, nullptr, 1, MPI_INT,
    //        rootPE.getRoot(), comm_);
    //    }

    //    // TODO Use correct type
    //    // TODO check if recvBuf is large enough
    //    MPI_Gatherv(
    //        sendBuf.get().ptr, mySendCount, MPI_INT, recvPtr, recvCountsPtr,
    //        recvDisplsPtr, MPI_INT, rootPE.getRoot(), comm_);

    //    return gatherv_output(std::move(recvBuf),
    //    std::move(recvCountsContainer), std::move(recvDisplsContainer));
    //}

    MPI_Comm get_comm() const {
        return comm_;
    }
    int rank() const {
        return rank_;
    }
    int size() const {
        return size_;
    }
    unsigned int rank_unsigned() const {
        return static_cast<unsigned int>(rank_);
    }
    unsigned int size_unsigned() const {
        return static_cast<unsigned int>(size_);
    }

private:
    void                         big_type_handling() const {}
    MPI_Comm                     comm_;
    int                          rank_;
    int                          size_;
    static constexpr std::size_t mpi_size_limit = std::numeric_limits<int>::max();
};
/// @}
} // namespace kamping
