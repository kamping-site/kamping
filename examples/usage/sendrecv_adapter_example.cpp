#include <iostream>
#include <mdspan>

#include "helpers_for_examples.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/data_buffers/pipe_db.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/sendrecv.hpp"

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    {
        auto   dest = comm.rank_shifted_cyclic(1);
        size_t size = 10;

        std::vector<int>                              sbuf(size * 5, comm.rank_signed() + 5);
        std::mdspan<int, std::extents<size_t, 5, 10>> ms_sbuf(sbuf.data());
        auto                                          rbuf = EmptyDataBuffer<int>();

        auto [sent, received] = comm.sendrecv(ms_sbuf | mdspan_adapter, rbuf, static_cast<int>(dest));

        if (comm.rank() == 0) {
            for (auto x: received) {
                print_on_root(std::to_string(x), comm);
            }
            print_on_root(std::to_string(std::ranges::size(sent)), comm);
        }
    }
}