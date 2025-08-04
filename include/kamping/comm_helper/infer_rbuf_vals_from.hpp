#include "kamping/communicator.hpp"

namespace kamping {

    struct resize_recv_count{};
    struct resize_send_size_allgather{};


    template <typename Buff>
    concept HasResize = requires(Buff buf, size_t count) {
        {buf.resize(count)};
    };

    template <typename Buff>
    concept HasCount = requires(Buff buf) {
        {buf.count()} -> std::integral<>;
    };

    template<
            template<typename...>
            typename DefaultContainerType,
            template<typename, template<typename...> typename>
            typename... Plugins>
    template<typename... Tags, typename SBuff, typename RBuff>
    void Communicator<DefaultContainerType, Plugins...>::infer_rbuf_vals_from(const SBuff& sbuf, RBuff& rbuf) const {
        size_t comm_size = size();
        (..., infer_rbuf_vals(Tags{}, sbuf, rbuf, comm_size));
    }

    template<typename SBuff, typename RBuff>
    requires HasResize<RBuff>
    void infer_rbuf_vals(resize_send_size_allgather, const SBuff& sbuf, RBuff& rbuf, size_t comm_size) {
        rbuf.resize(sbuf.size() * comm_size);
        std::cout << "Resized to sbuf.size() * comm_size" << std::endl;
    }

    template<typename SBuff, typename RBuff>
    requires HasResize<RBuff> && HasCount<RBuff>
    void infer_rbuf_vals(resize_recv_count, const SBuff& sbuf, RBuff& rbuf, size_t comm_size) {
        rbuf.resize(rbuf.count());
        std::cout << "Resized to rbuf.count()" << std::endl;
    }
}