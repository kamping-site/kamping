#pragma once

namespace kamping {

enum class CommType { allgather, gather, recv };

template <typename Buff>
concept HasTag = requires(Buff buf) {
    { buf.tag() } -> std::integral<>;
};

template <typename Buff>
concept HasSource = requires(Buff buf) {
    { buf.source() } -> std::integral<>;
};

template <typename Buff>
concept HasStatus = requires(Buff buf) {
    {buf.status()};
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
