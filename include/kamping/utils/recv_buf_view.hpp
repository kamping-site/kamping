#pragma once

#include <kamping/span.hpp>

namespace kamping {

template <typename T>
struct recv_buf_view {
    Span<T>   recv_buf;
    Span<int> recv_counts;
    Span<int> recv_displs;

    template <typename RecvBuf, typename RecvCounts, typename RecvDispls>
    recv_buf_view(RecvBuf const& recv_buf, RecvCounts const& recv_counts, RecvDispls const& recv_displs)
        : recv_buf(recv_buf),
          recv_counts(recv_counts),
          recv_displs(recv_displs) {}

    Span<T> operator[](int i) const {
        return recv_buf.subspan(recv_displs[i], recv_counts[i]);
    }
};

} // namespace kamping
