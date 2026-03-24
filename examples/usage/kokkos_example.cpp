#include <iostream>

#include <KokkosComm/KokkosComm.hpp>
#include <Kokkos_Random.hpp>

#include "kamping/communicator.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/sendrecv.hpp"

template <class ViewType>
class KokkosViewAdaptor {
public:

    explicit KokkosViewAdaptor(ViewType& view) : _view(view) {}

    auto type() {
        return kamping::mpi_datatype<typename ViewType::non_const_value_type>();
    }

    auto begin() {
        return data();
    }
    auto end() {
        KAMPING_ASSERT(_view.span_is_contiguous(), "View is not contiguous");
        return data() + _view.size();
    }

    auto data() {
        KAMPING_ASSERT(_view.span_is_contiguous(), "View is not contiguous");
        return _view.data();
    }

    auto size() {
        return _view.size();
    }

    void resize(size_t size) {
        Kokkos::resize(_view, size);
    }

private:
    ViewType& _view;
};

template <typename DataType, typename... Properties>
void print_view(const char* prefix, const Kokkos::View<DataType, Properties...>& view) {
    auto view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);


    KAMPING_ASSERT(view.span_is_contiguous(), "View is not contiguous");
    std::string output = prefix;
    auto* ptr = view_host.data();
    for (size_t i = 0; i < view_host.size(); ++i) {
        output += "[" + std::to_string(ptr[i]) + "], ";
    }
    std::cout << output << std::endl;
}

int main(int argc, char **argv) {
    kamping::Environment e;
    kamping::Communicator comm;

    Kokkos::initialize(argc, argv);
    {
        int rank, size;
        MPI_Comm_rank(comm.mpi_communicator(), &rank);
        MPI_Comm_size(comm.mpi_communicator(), &size);

        using Scalar = double;

        using Mode      = KokkosComm::mpi::DefaultCommMode;
        auto space      = Kokkos::DefaultExecutionSpace();
        using view_type = Kokkos::View<Scalar *>;

        {
            if (rank == 0) std::cout << "Using Kokkos comm: " << std::endl;
            view_type to_send("", 10);
            Kokkos::Random_XorShift64_Pool<> random_pool(static_cast<std::uint64_t>(rank));
            Kokkos::fill_random(to_send, random_pool, 0.0, 1.0);

            // Will just throw MPI_ERR_TRUNCATE if to_recv is too small
            view_type to_recv("", 10);


            if (0 == rank) {
                print_view("Before send on 0: ", to_send);
                KokkosComm::mpi::send(space, to_send, 1, 0, comm.mpi_communicator(), Mode{});

                KokkosComm::mpi::recv(space, to_recv, 1, 0, comm.mpi_communicator());
                print_view("After recv on 0: ", to_recv);
            } else if (1 == rank) {
                KokkosComm::mpi::recv(space, to_recv, 0, 0, comm.mpi_communicator());
                print_view("After recv on 1: ", to_recv);

                print_view("Before send on 1: ", to_send);
                KokkosComm::mpi::send(space, to_send, 0, 0, comm.mpi_communicator(), Mode{});

            }
        }


        {
            if (rank == 0) std::cout << "Using kamping with kokkos view: " << std::endl;

            Kokkos::View<double**> to_send ("", 3, 3);
            Kokkos::Random_XorShift64_Pool<> random_pool(static_cast<std::uint64_t>(rank));
            Kokkos::fill_random(to_send, random_pool, 0.0, 1.0);

            view_type to_recv("", 1);

            KokkosViewAdaptor send_adaptor(to_send);
            KokkosViewAdaptor recv_adaptor(to_recv);

            if (rank == 0) {
                print_view("Before send on 0: ", to_send);
                comm.sendrecv(send_adaptor, recv_adaptor | resize_buf(), 1);
                print_view("After recv on 0: ", to_recv);
            }
            else if (rank == 1) {
                print_view("Before send on 1: ", to_send);
                comm.sendrecv(send_adaptor, recv_adaptor | resize_buf(), 0);
                print_view("After recv on 1: ", to_recv);
            }
        }

        Kokkos::finalize();
        return 0;
    }
}

