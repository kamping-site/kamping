#pragma once

#include "kamping/mpi_datatype.hpp"

namespace kamping {

enum class CommType { allgather, gather, recv, sendrecv, alltoall, alltoallv };

template <typename Buff>
concept HasSetSize = requires(Buff buf, size_t size) {
    {buf.set_size(size)};
};

template <typename Buff>
concept HasSetSizeV = requires(Buff buf, std::vector<int>&& sizes) {
    {buf.set_size_v(std::move(sizes))};
};

template <typename Buff>
concept HasSetDisplacements = requires(Buff buf, std::vector<int>&& displacements) {
    {buf.set_displacements(std::move(displacements))};
};

template <typename T>
concept static_mpi_type = has_static_type_v<T>;

template <typename Buff>
concept HasType = requires(Buff buf) {
    { buf.type() } -> std::same_as<MPI_Datatype>;
};

template <typename Buff>
requires HasType<Buff> || static_mpi_type<std::ranges::range_value_t<Buff>> MPI_Datatype type(Buff& buf) {
    if constexpr (HasType<Buff>) {
        return buf.type();
    } else if constexpr (static_mpi_type<std::ranges::range_value_t<Buff>>) {
        return mpi_datatype<std::ranges::range_value_t<Buff>>();
    }
}

template <typename Buff>
concept Typed = requires(Buff buf) {
    { type(buf) } -> std::same_as<MPI_Datatype>;
};

// Returns the size of the total communication. E.g. the size that the receiving buffer needs to be
template <CommType commType, typename SBuff, typename RBuff, typename Communicator>
size_t get_recv_size(SBuff const& sbuf, RBuff& rbuf, Communicator const& comm) {
    using send_type = decltype(type(sbuf));
    using recv_type = decltype(type(rbuf));

    if constexpr (commType == CommType::allgather) {
        if constexpr (std::is_same_v<send_type, recv_type>) {
            return asserting_cast<size_t>(std::ranges::size(sbuf) * comm.size());
        } else {
            return asserting_cast<size_t>(std::ranges::size(rbuf));
        }
    }

    if constexpr (commType == CommType::alltoall) {
        if constexpr (std::is_same_v<send_type, recv_type>) {
            return asserting_cast<size_t>(std::ranges::size(sbuf));
        } else {
            return asserting_cast<size_t>(std::ranges::size(rbuf));
        }
    }

    return 0;
}

} // namespace kamping
