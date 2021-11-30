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

    template<class recvBuffType>
    struct gatherv_output {
        gatherv_output(recvBuffType &&recvBuff) : _recvBuff(recvBuff) {}

        // TODO Try to do this by checking whether extract() exists
        template<typename recvBuffType2 = recvBuffType, std::enable_if_t<recvBuffType2::isExtractable, bool> = true>
        decltype(auto) getRecvBuff() {
            return _recvBuff.extract();
        }

        // TODO add recv counts as optional output
        // TODO add recv displacements as optional output

    private:
        recvBuffType _recvBuff;
    };

    template<class... Args>
    auto gatherv(Args &&... args) {
        auto sendBuf = select_trait<ptraits::in>(std::forward<Args>(args)...);
        using send_type = typename decltype(sendBuf)::value_type;

        // Get receive buffer and use new vector if none is given
        auto recvBuf = select_trait<ptraits::out>(std::forward<Args>(args)..., out(new_vector<send_type>()));
        using recv_type = typename decltype(recvBuf)::value_type;

        // TODO add recv counts as optional input parameter
        // TODO add recv displacements as optional input parameter

        // Select root. Defaults to 0
        // TODO let use choose default root for context
        auto rootPE = select_trait<ptraits::root>(std::forward<Args>(args)..., root(0));

        //  Gather send counts at root
        // TODO don't do this if the user supplies send counts at root
        // TODO only do these allocations and calculations on root
        int mySendCount = sendBuf.get().size;
        std::vector<int> recvCounts(size_);

        MPI_Gather(&mySendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, rootPE.getRoot(), comm_);

        std::vector<int> recvDispls(size_);
        std::exclusive_scan(recvCounts.begin(), recvCounts.end(), recvDispls.begin(), 0);

        int recvSize = recvDispls.back() + recvCounts.back();
        auto recvPtr = recvBuf.get_ptr(recvSize);

        // TODO Use correct type
        MPI_Gatherv(sendBuf.get().ptr,
                    mySendCount,
                    MPI_INT,
                    recvPtr,
                    recvCounts.data(),
                    recvDispls.data(),
                    MPI_INT,
                    rootPE.getRoot(),
                    comm_);

        return gatherv_output(std::move(recvBuf));
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
