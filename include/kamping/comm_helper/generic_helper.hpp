#pragma once

namespace kamping {

enum class CommType { allgather, gather, recv };

template <typename Buff>
concept HasSetSize= requires(Buff buf, size_t size) {
    { buf.set_size(size) };
};


// Returns the size of the total communication. E.g. the size that the receiving buffer needs to be
template <CommType type, typename SBuff, typename RBuff, typename Communicator>
size_t get_recv_size(SBuff const& sbuf, RBuff& rbuf, Communicator const& comm) {
    using send_type = std::ranges::range_value_t<SBuff>;
    using recv_type = std::ranges::range_value_t<RBuff>;

    if constexpr (type == CommType::allgather) {
        if constexpr (std::is_same_v<send_type, recv_type>) {
            return asserting_cast<size_t>(std::size(sbuf) * comm.size());
        } else {
            return asserting_cast<size_t>(std::size(rbuf));
        }
    }

    return 0;
}

} // namespace kamping
