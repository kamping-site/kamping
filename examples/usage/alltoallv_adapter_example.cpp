#include <iostream>
#include <mdspan>

#include "helpers_for_examples.hpp"
#include "kamping/adapter/mdspan_adapter.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
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

        auto set_displs = add_displs() | auto_displs();
        auto send_buf   = adapter::MDSpanAdapter(ms_sbuf) | add_size_v(size_v) | set_displs;

        std::vector<int>                                        rbuf(size * num_elems, 0);
        std::mdspan<int, std::extents<size_t, size, num_elems>> ms_rbuf(rbuf.data());
        std::vector<int>                                        size_v_recv(size, num_elems);

        // Does not satisfy HasDispls
        // auto a = adapter::MDSpanAdapter(ms_rbuf) | auto_displs();

        // Does not satisfy HasSizeV
        // auto a = adapter::MDSpanAdapter(ms_rbuf) | add_displs() | auto_displs();
        // auto a = adapter::MDSpanAdapter(ms_rbuf) | add_displs() | auto_displs() | add_size_v(size_v_recv);

        // add_size_v inbetween works
        // auto recv_buf = adapter::MDSpanAdapter(ms_rbuf) | add_displs() | add_size_v(size_v_recv) | auto_displs() ;

        auto recv_buf     = adapter::MDSpanAdapter(ms_rbuf) | add_size_v(size_v_recv) | add_displs() | auto_displs();
        auto [sent, recv] = comm.alltoallv(send_buf, recv_buf);

        if (comm.rank_signed() == 0) {
            std::cout << "md_span sent:" << std::endl;
            auto md_sent = sent.get_base().get_mdspan();
            print_md_span(md_sent);

            std::cout << "md_span received:" << std::endl;
            auto md_recv = recv.get_base().get_mdspan();
            print_md_span(md_recv);
        }
    }
}