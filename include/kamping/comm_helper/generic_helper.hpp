#pragma once

namespace kamping{

    template <typename Buff>
    concept HasResize = requires(Buff buf, size_t count) {
        {buf.resize(count)};
    };

    template <typename Buff>
    concept HasUnderlying = requires(Buff buf) {
        {buf.underlying()};
    };


    template <typename Buff>
    concept HasCount = requires(Buff buf) {
        {buf.count()} -> std::integral<>;
    };

    enum class CommType {
        allgather,
        gather
    };

// Returns the count of the communication. E.g. the count used in the MPI call
template <CommType type, typename SBuff, typename RBuff, typename Communicator>
size_t get_recv_count(const SBuff& sbuf, RBuff& rbuf, const Communicator&) {

    using send_type = std::ranges::range_value_t<SBuff>;
    using recv_type = std::ranges::range_value_t<RBuff>;

    if constexpr (type == CommType::allgather) {
        if constexpr (std::is_same_v<send_type, recv_type>) {
           return asserting_cast<size_t>(sbuf.size());
        } else {
            static_assert(HasCount<RBuff>, "Recv type differs from send type and no recv count is given");
            return asserting_cast<size_t>(rbuf.count());
        }
    }
    return 0;
}

// Returns the size of the total communication. E.g. the size that the receiving buffer needs to be
template <CommType type, typename SBuff, typename RBuff, typename Communicator>
size_t get_recv_size(const SBuff& sbuf, RBuff& rbuf, const Communicator& comm) {

    using send_type = std::ranges::range_value_t<SBuff>;
    using recv_type = std::ranges::range_value_t<RBuff>;

    if constexpr (type == CommType::allgather) {
        if constexpr (std::is_same_v<send_type, recv_type>) {
            return asserting_cast<size_t>(sbuf.size() * comm.size());
        } else {
            static_assert(HasCount<RBuff>, "Recv type differs from send type and no recv count is given");
            return asserting_cast<size_t>(rbuf.count());
        }
    }
    return 0;
}





}
