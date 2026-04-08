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
#include "kamping/v2/p2p/isend.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
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
        kamping::v2::send(v, 1, comm);
    } else if (comm.rank() == 1) {
        std::vector<int> v;
        kamping::v2::recv(v | kamping::views::resize);
        std::println("result = {}", v);
    }
    // std::vector v = {1, 2, 3};
    if (comm.rank() == 0) {
        std::unordered_map<std::string, int> map{{"one", 1}, {"two", 2}, {"forty-two", 42}};
        kamping::v2::send(map | kamping::views::serialize, 1, comm);
    } else if (comm.rank() == 1) {
        auto result = kamping::v2::recv(kamping::views::deserialize<std::unordered_map<std::string, int>>(), comm);
        std::println("result = {}", *result);
    }

    if (comm.rank() == 0) {
        std::unordered_map<std::string, int> map{{"ett", 1}, {"två", 2}, {"fyrtio-två", 42}};
        MPI_Request                          request = MPI_REQUEST_NULL;
        kamping::v2::isend(kamping::v2::send_mode::standard, &request, map | kamping::views::serialize, 1);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    } else if (comm.rank() == 1) {
        auto result = kamping::v2::recv(kamping::views::deserialize<std::unordered_map<std::string, int>>());
        std::println("result = {}", *result);
    }
    return 0;
}
