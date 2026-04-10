#include <cstddef>
#include <print>
#include <vector>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/v2/contrib/cereal_view.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/irecv.hpp"
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/p2p/isendrecv.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/p2p/sendrecv.hpp"
#include "kamping/v2/views/resize_view.hpp"

template <>
struct kamping::bridge::native_handle_traits<kamping::Communicator<>> {
    static MPI_Comm handle(kamping::Communicator<> const& comm) {
        return comm.mpi_communicator();
    }
};

struct my_struct {
    int val;
};

template <>
struct kamping::ranges::buffer_traits<my_struct> {
    static std::ptrdiff_t size(my_struct const&) {
        return 1;
    }
    static int const* data(my_struct const& t) {
        return &t.val;
    }
    static int* data(my_struct& t) {
        return &t.val;
    }
    static MPI_Datatype type(my_struct const&) {
        return MPI_INT;
    }
};

int main(int, char*[]) {
    kamping::Environment<>  env;
    kamping::Communicator<> comm;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    kamping::core::send(my_struct{}, MPI_PROC_NULL, 0, MPI_COMM_WORLD);
    if (comm.rank() == 0) {
        std::vector<int> v{1, 2, 3, 4};
        kamping::v2::send(std::move(v), 1, comm);
    } else if (comm.rank() == 1) {
        std::vector<int> v;
        kamping::v2::recv(v | kamping::views::resize);
        std::println("result = {}", v);
    }

    if (comm.rank() == 0) {
        std::unordered_map<std::string, int> map{{"one", 1}, {"two", 2}, {"forty-two", 42}};
        kamping::v2::send(map | kamping::views::serialize, 1, comm);
    } else if (comm.rank() == 1) {
        auto result = kamping::v2::recv(kamping::views::deserialize<std::unordered_map<std::string, int>>(), comm);
        std::println("result = {}", *result);
    }
    if (comm.rank() == 0) {
        std::vector<int> const v{11, 12, 13, 14};
        kamping::v2::isend(v, 1).wait();
    } else if (comm.rank() == 1) {
      MPI_Status status;
      auto       v = kamping::v2::irecv(std::vector<int>{10} | kamping::views::resize).wait(&status);
      std::println("v = {}", v);
    }

    if (comm.rank() == 0) {
        std::unordered_map<std::string, int> map{{"ett", 1}, {"två", 2}, {"fyrtio-två", 42}};
        kamping::v2::isend(map | kamping::views::serialize, 1).wait();
    } else if (comm.rank() == 1) {
        auto result = kamping::v2::recv(kamping::views::deserialize<std::unordered_map<std::string, int>>());
        std::println("result = {}", *result);
    }

    // sendrecv: each rank simultaneously sends to the other and receives from the other
    if (comm.size() >= 2 && (comm.rank() == 0 || comm.rank() == 1)) {
        int const         peer = 1 - static_cast<int>(comm.rank());
        std::vector<int>  send_data = (comm.rank() == 0) ? std::vector<int>{1, 2, 3} : std::vector<int>{4, 5, 6};
        std::vector<int>  recv_data;
        auto&& [_, recvd] = kamping::v2::sendrecv(send_data, peer, recv_data | kamping::views::resize, peer, comm);
        std::println("rank {} recvd = {}", comm.rank(), recvd);
    }

    // isendrecv: non-blocking sendrecv, wait() to retrieve the result
    if (comm.size() >= 2 && (comm.rank() == 0 || comm.rank() == 1)) {
        int const        peer = 1 - static_cast<int>(comm.rank());
        std::vector<int> send_data = (comm.rank() == 0) ? std::vector<int>{7, 8, 9} : std::vector<int>{10, 11, 12};
        std::vector<int> recv_data;
        auto&& [_2, recvd2] =
            kamping::v2::isendrecv(send_data, peer, recv_data | kamping::views::resize, peer, comm).wait();
        std::println("rank {} isendrecv recvd = {}", comm.rank(), recvd2);
    }
    return 0;
}
