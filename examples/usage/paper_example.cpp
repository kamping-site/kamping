#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <mpi.h>

#include "kamping/named_parameters.hpp"

struct MyType {
    int                a;
    double             b;
    char               c;
    std::array<int, 3> d;
};

namespace kamping {
// using KaMPIng’s built-in struct serializer
template <>
struct mpi_type_traits<MyType> : struct_type<MyType> {};
// or using an explicitly constructed type
// template <>
// struct mpi_type_traits<MyType2> {
// static constexpr bool has_to_be_committed = true;
// static MPI_Datatype data_type() {
// MPI_Datatype type;
// MPI_Type_create_*(..., &type);
// return type;
// }
// };
} // namespace kamping

template <typename T>
auto build_buckets(std::vector<T>& data, std::vector<T>& splitters) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> buckets(splitters.size() + 1);
    for (auto& element: data) {
        auto const bound = std::upper_bound(splitters.begin(), splitters.end(), element);
        buckets[bound - splitters.begin()].push_back(element);
    }
    data.clear();
    return buckets;
}

// Sorting code for Fig. 7
template <typename T>
void sort(std::vector<T>& data, MPI_Comm comm_) {
    using namespace std;
    using namespace kamping;
    Communicator comm(comm_);
    size_t const num_samples = 16 * log2(comm.size()) + 1;
    vector<T>    lsamples(num_samples);
    sample(data.begin(), data.end(), lsamples.begin(), num_samples, mt19937{random_device{}()});
    auto gsamples = comm.allgather(send_buf(lsamples));
    sort(gsamples.begin(), gsamples.end());
    for (size_t i = 0; i < comm.size() - 1; i++) {
        gsamples[i] = gsamples[num_samples * (i + 1)];
    }
    gsamples.resize(comm.size() - 1);
    vector<vector<T>> buckets = build_buckets(data, gsamples);
    data.clear();
    vector<int> scounts;
    for (auto& bucket: buckets) {
        data.insert(data.end(), bucket.begin(), bucket.end());
        scounts.push_back(bucket.size());
    }
    data = comm.alltoallv(send_buf(data), send_counts(scounts));
    sort(data.begin(), data.end());
}

// These are the examples from the paper. Some examples are not runnable as is, but everything should compile.
// If some change breaks any of these, consider updating the arxiv paper.
auto main() -> int {
    kamping::Environment  env;
    kamping::Communicator comm;

    using namespace kamping;
    {
        // Fig. 1.
        std::vector<double> v = {0.1, 3.14, 4.2, 123.4};
        {
            // KaMPIng allows concise code
            // with sensible defaults ... (1)

            auto v_global = comm.allgatherv(send_buf(v));
        }
        {
            // ... or detailed tuning of each parameter (2)
            std::vector<int> rc;
            auto [v_global, rcounts, rdispls] = comm.allgatherv(
                send_buf(v),                                           //(3)
                recv_counts_out<resize_to_fit /*(6)*/>(std::move(rc)), //(4)
                recv_displs_out()                                      //(5)
            );
        }
    }
    {
        // Fig. 2.
        using T                 = int;
        MPI_Datatype   MPI_TYPE = MPI_INT;
        MPI_Comm       comm     = MPI_COMM_WORLD;
        std::vector<T> v        = {1, 3, 4}; // fill with data
        int            size, rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        std::vector<int> rc(size), rd(size);
        rc[rank] = v.size();
        // exchange counts
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rc.data(), 1, MPI_INT, comm);
        // compute displacements
        std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
        int n_glob = rc.back() + rd.back();
        // allocate receive buffer
        std::vector<T> v_glob(n_glob);
        // exchange
        MPI_Allgatherv(v.data(), v.size(), MPI_TYPE, v_glob.data(), rc.data(), rd.data(), MPI_TYPE, comm);
    }
    {
        // Fig. 3.
        std::vector<int> v = {1, 3, 4}; // fill with data
        using T            = int;

        {
            // Version 1: using KaMPIng’s interface
            std::vector<int> rc(comm.size()), rd(comm.size());
            rc[comm.rank()] = v.size();
            comm.allgather(send_recv_buf(rc));
            std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
            std::vector<T> v_glob(rc.back() + rd.back());
            comm.allgatherv(send_buf(v), recv_buf(v_glob), recv_counts(rc), recv_displs(rd));
        }
        {
            // Version 2: displacements are computed implicitly
            std::vector<int> rc(comm.size());
            rc[comm.rank()] = v.size();
            comm.allgather(send_recv_buf(rc));
            std::vector<T> v_glob;
            comm.allgatherv(send_buf(v), recv_buf<resize_to_fit>(v_glob), recv_counts(rc));
        }
        {
            // Version 3: counts are automatically exchanged
            // and result is returned by value
            std::vector<T> v_glob = comm.allgatherv(send_buf(v));
        }
    }
    {
        std::vector<int> v = {1, 3, 4}; // fill with data

        // Section III snippets
        {
            auto result   = comm.allgatherv(send_buf(v), recv_counts_out());
            auto recv_buf = result.extract_recv_buf();
            auto counts   = result.extract_recv_counts();
        }
        { //
            auto [recv_buf, counts] = comm.allgatherv(send_buf(v), recv_counts_out());
        }
        {
            using T            = int;
            std::vector<T> tmp = {1, 2, 3, 4};
            // tmp is moved to the underlying call where the
            // storage is reused for the recv buffer
            auto recv_buffer = comm.allgatherv(send_buf(v), recv_buf(std::move(tmp)));
        }
        {
            using T                    = int;
            std::vector<T> recv_buffer = {};
            // data is written into recv_buffer directly
            comm.allgatherv(send_buf(v), recv_buf(recv_buffer));
        }
        {
            using T = int;
            std::vector<T>   recv_buffer;         // has to be resized
            std::vector<int> counts(comm.size()); // size large enough
            comm.allgatherv(send_buf(v), recv_buf<resize_to_fit>(recv_buffer), recv_counts_out(counts));
        }
    }
    {
        // Fig. 4.
        // type definition is on top
        MyType x{};
        comm.send(send_buf(x), destination(rank::null));
    }
    {
        // Fig. 5.
        using dict = std::unordered_map<std::string, std::string>;
        dict data  = {{"foo", "bar"}, {"baz", "x"}};
        comm.send(send_buf(as_serialized(data)), destination(rank::null));
        dict recv_dict = comm.recv(recv_buf(as_deserializable<dict>()));
    }
    {
        // Fig. 6.
        std::vector<int> v  = {1, 3, 5};
        auto             r1 = comm.isend(send_buf_out(std::move(v)), destination(1));
        v                   = r1.wait(); // v is moved back to caller after
        // request is complete
        auto             r2   = comm.irecv<int>(recv_count(42));
        std::vector<int> data = r2.wait(); // data only returned
                                           // after request
                                           // is complete
    }
    {
        // Sec. III.G snippet
        std::vector<int> data(comm.size());
        data[comm.rank()] = comm.rank();
        data              = comm.allgather(send_recv_buf(std::move(data)));
    }
    {
        // Fig. 7.
        std::vector<int> data = {13, 1, 7, 18};
        sort(data, MPI_COMM_WORLD);
    }
    return 0;
}
