#include <iostream>
#include <mdspan>

#include "helpers_for_examples.hpp"
#include "kamping/adapter/mdspan_adapter.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"
#include "kamping/environment.hpp"

template <typename span>
void print_md_span(span mdspan) {
    for (size_t i = 0; i < mdspan.extent(0); ++i) {
        for (size_t j = 0; j < mdspan.extent(1); ++j) {
            std::cout << mdspan[i, j] << ' ';
        }
        std::cout << '\n';
    }
}

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    if (comm.size() != 3) {
        return 0;
    }

    {
        const size_t size      = 3;
        const size_t num_elems = 10;

        std::vector<int>                                        sbuf(size * num_elems, comm.rank_signed() * 10);
        std::mdspan<int, std::extents<size_t, size, num_elems>> ms_sbuf(sbuf.data());
        std::vector<int>                                        size_v(size, num_elems);

        auto send_buf = adapter::MDSpanAdapter(ms_sbuf) | with_size_v(size_v) | auto_displs();

        std::vector<int>                                        rbuf(size * num_elems, 0);
        std::mdspan<int, std::extents<size_t, size, num_elems>> ms_rbuf(rbuf.data());
        std::vector<int>                                        size_v_recv(size, num_elems);

        auto recv_buf     = adapter::MDSpanAdapter(ms_rbuf) | with_size_v(size_v_recv) | auto_displs();
        auto [sent, recv] = comm.alltoallv(send_buf, recv_buf);

        if (comm.rank_signed() == 0) {
            std::cout << "md_span sent:" << std::endl;
            auto md_sent = sent.buffer().get_mdspan();
            print_md_span(md_sent);

            std::cout << "md_span received:" << std::endl;
            auto md_recv = recv.buffer().get_mdspan();
            print_md_span(md_recv);
        }
    }
}