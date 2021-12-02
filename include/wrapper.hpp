#pragma once

#include "mpi_ops.hpp"
#include "trait_selection.hpp"
#include "type_helpers.hpp"

#include <limits>
#include <mpi.h>
#include <numeric>
#include <type_traits>
#include <vector>


namespace MPIWrapper {

struct Rank {
    int rank;
};

class MPIContext {
public:
    enum class SendMode { normal, buffered, synchronous };
    explicit MPIContext(MPI_Comm comm) : comm_{comm} {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }

    template<class recvBuffType, class recvCountsType, class recvDisplsType>
    struct gatherv_output {
        gatherv_output(recvBuffType &&recvBuff, recvCountsType &&recvCounts, recvDisplsType &&recvDispls)
            : _recvBuff(std::move(recvBuff))
            , _recvCounts(std::move(recvCounts))
            , _recvDispls(std::move(recvDispls)) {}

        // TODO Try to do this by checking whether extract() exists
        template<typename recvBuffType2 = recvBuffType, std::enable_if_t<recvBuffType2::isExtractable, bool> = true>
        decltype(auto) extractRecvBuff() {
            return _recvBuff.extract();
        }

        template<typename recvCountsType2 = recvCountsType, std::enable_if_t<recvCountsType2::isExtractable, bool> = true>
        decltype(auto) extractRecvCounts() {
            return _recvCounts.extract();
        }


        template<typename recvDisplsType2 = recvDisplsType, std::enable_if_t<recvDisplsType2::isExtractable, bool> = true>
        decltype(auto) extractRecvDispls() {
            return _recvDispls.extract();
        }

    private:
        recvBuffType _recvBuff;
        recvCountsType _recvCounts;
        recvDisplsType _recvDispls;
    };

    template<class... Args>
    auto gatherv(Args &&... args) {
        auto sendBuf = select_trait<ptraits::in>(std::forward<Args>(args)...);
        using send_type = typename decltype(sendBuf)::value_type;

        // Get receive buffer and use new vector if none is given
        auto recvBuf = select_trait<ptraits::out>(std::forward<Args>(args)..., out(new_vector<send_type>()));
        using recv_type = typename decltype(recvBuf)::value_type;

        auto recvCountsContainer =
            select_trait<ptraits::recvCounts>(std::forward<Args>(args)..., recv_counts(new_vector<int>()));
        static_assert(std::is_same<typename decltype(recvCountsContainer)::value_type, int>::value,
                      "Recv counts must be int");
        auto recvCountsPtr = recvCountsContainer.get_ptr(size_);

        auto recvDisplsContainer =
            select_trait<ptraits::recvDispls>(std::forward<Args>(args)..., recv_displs(new_vector<int>()));
        static_assert(std::is_same<typename decltype(recvDisplsContainer)::value_type, int>::value,
                      "Recv displacements must be int");
        auto recvDisplsPtr = recvDisplsContainer.get_ptr(size_);

        // Select root. Defaults to 0
        // TODO let use choose default root for context
        auto rootPE = select_trait<ptraits::root>(std::forward<Args>(args)..., root(0));

        //  Gather send counts at root
        // TODO don't do this if the user supplies send counts at root
        // TODO only do these allocations and calculations on root
        int mySendCount = sendBuf.get().size;

        MPI_Gather(&mySendCount, 1, MPI_INT, recvCountsPtr, 1, MPI_INT, rootPE.getRoot(), comm_);

        std::exclusive_scan(recvCountsPtr, recvCountsPtr + size_, recvDisplsPtr, 0);

        int recvSize = *(recvDisplsPtr + size_ - 1) + *(recvCountsPtr + size_ - 1);
        auto recvPtr = recvBuf.get_ptr(recvSize);

        // TODO Use correct type
        MPI_Gatherv(sendBuf.get().ptr, mySendCount, MPI_INT, recvPtr, recvCountsPtr, recvDisplsPtr, MPI_INT, rootPE.getRoot(), comm_);

        return gatherv_output(std::move(recvBuf), std::move(recvCountsContainer), std::move(recvDisplsContainer));
    }

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
    void big_type_handling() const {}
    MPI_Comm comm_;
    int rank_;
    int size_;
    static constexpr std::size_t mpi_size_limit = std::numeric_limits<int>::max();
};
} // namespace MPIWrapper
