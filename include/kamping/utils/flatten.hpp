#pragma once
#include <numeric>
#include <utility>
#include <vector>

#include "kamping/named_parameters.hpp"

template <typename F>
struct CallableDoWrapper {
    F f;
    template <typename... Args>
    auto do_(Args&&... args) {
        return f(std::forward<Args...>(args)...);
    }
};

template <typename F>
auto make_callable_do_wrapper(F f) {
    return CallableDoWrapper<F>{std::move(f)};
}

namespace kamping {
template <
    template <typename...> typename CountContainer = std::vector,
    typename SparseSendBuf,
    typename CommunicatorType>
auto with_flattened(SparseSendBuf const& sparse_send_buf, CommunicatorType const& comm) {
    CountContainer<int> send_counts(comm.size());
    for (auto const& [destination, message]: sparse_send_buf) {
        auto send_buf                                    = kamping::send_buf(message);
        send_counts[asserting_cast<size_t>(destination)] = asserting_cast<int>(send_buf.size());
    }
    CountContainer<int> send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    size_t total_send_count = asserting_cast<size_t>(send_displs.back() + send_counts.back());
    using FlatContainer     = std::remove_const_t<std::tuple_element_t<1, typename SparseSendBuf::value_type>>;
    FlatContainer flat_send_buf;
    flat_send_buf.resize(total_send_count);
    for (auto const& [destination, message]: sparse_send_buf) {
        auto send_buf = kamping::send_buf(message).construct_buffer_or_rebind();
        std::copy_n(
            send_buf.data(),
            send_buf.size(),
            flat_send_buf.data() + send_displs[asserting_cast<size_t>(destination)]
        );
    }
    return make_callable_do_wrapper([flat_send_buf = std::move(flat_send_buf),
                                     send_counts   = std::move(send_counts),
                                     send_displs   = std::move(send_displs)](auto&& f) {
        return std::apply(
            std::forward<decltype(f)>(f),
            std::tuple(
                kamping::send_buf(flat_send_buf),
                kamping::send_counts(send_counts),
                kamping::send_displs(send_displs)
            )
        );
    });
}
} // namespace kamping
