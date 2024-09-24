// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.
#include <algorithm>
#include <random>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/utils/flatten.hpp>
#include <mpi.h>

#include "kamping/collectives/allgather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

using namespace ::testing;

namespace sorting {
template <typename T>
auto build_buckets(std::vector<T>& data, std::vector<T>& splitters) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> buckets(splitters.size() + 1);
    for (auto& element: data) {
        auto const bound = std::upper_bound(splitters.begin(), splitters.end(), element);
        buckets[static_cast<size_t>(bound - splitters.begin())].push_back(element);
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
    size_t const num_samples = static_cast<size_t>(16 * log2(comm.size()) + 1);
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
        scounts.push_back(static_cast<int>(bucket.size()));
    }
    data = comm.alltoallv(send_buf(data), send_counts(scounts));
    sort(data.begin(), data.end());
}
} // namespace sorting

namespace bfs {
struct Graph {
    using Edges = std::vector<std::pair<size_t, int>>;
    bool is_local(size_t v) const {
        return v_begin <= v && v < v_end;
    }
    size_t local_id(size_t v) const {
        return v - v_begin;
    }
    size_t local_size() const {
        return v_end - v_begin;
    }
    Edges const& neighbors(size_t local_v) const {
        return edges[local_v];
    }
    size_t             v_begin;
    size_t             v_end;
    std::vector<Edges> edges;
};

Graph init_graph() {
    kamping::Communicator comm;
    Graph                 g;
    if (comm.rank() == 0) {
        g.v_begin = 0;
        g.v_end   = 2;
        g.edges.resize(2);
        g.edges[0].emplace_back(1, 0);
        g.edges[1].emplace_back(0, 0);
        g.edges[1].emplace_back(2, 1);
    }
    if (comm.rank() == 1) {
        g.v_begin = 2;
        g.v_end   = 4;
        g.edges.resize(2);
        g.edges[0].emplace_back(1, 0);
        g.edges[0].emplace_back(3, 1);
        g.edges[1].emplace_back(2, 1);
        g.edges[1].emplace_back(4, 2);
    }
    if (comm.rank() == 2) {
        g.v_begin = 4;
        g.v_end   = 6;
        g.edges.resize(2);
        g.edges[0].emplace_back(3, 1);
        g.edges[0].emplace_back(5, 2);
        g.edges[1].emplace_back(4, 2);
        g.edges[1].emplace_back(6, 3);
    }
    if (comm.rank() == 3) {
        g.v_begin = 6;
        g.v_end   = 8;
        g.edges.resize(2);
        g.edges[0].emplace_back(5, 2);
        g.edges[0].emplace_back(7, 3);
        g.edges[1].emplace_back(6, 3);
        g.edges[1].emplace_back(0, 0);
    }
    return g;
}

using namespace kamping;
using VId           = size_t;
using VBuf          = std::vector<VId>;
constexpr VId undef = std::numeric_limits<VId>::max();

bool is_empty(VBuf const& frontier, Communicator<> const& comm) {
    return comm.allreduce_single(send_buf(frontier.empty()), op(std::logical_and<>{}));
}

// changed signature from auto to std::unordered_map to be C++17 compliant
VBuf exchange(std::unordered_map<int, VBuf> frontier, Communicator<> const& comm) {
    return with_flattened(frontier, comm.size()).call([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)...);
    });
}

// not part of listing
std::unordered_map<int, VBuf>
expand_frontier(Graph const& graph, size_t level, VBuf const& frontier, std::vector<size_t>& dist) {
    std::unordered_map<int, VBuf> next_frontier;
    for (auto const& v: frontier) {
        auto  v_local  = graph.local_id(v);
        auto& cur_dist = dist[graph.local_id(v)];
        if (cur_dist == undef) {
            cur_dist = level;
            for (auto const& [u, rank]: graph.neighbors(v_local)) {
                next_frontier[rank].push_back(u);
            }
        }
    }
    return next_frontier;
}

std::vector<size_t> bfs(Graph const& g, VId s, MPI_Comm _comm) {
    Communicator                  comm(_comm);
    VBuf                          frontier;
    std::unordered_map<int, VBuf> next_frontier;
    std::vector<size_t>           dist(g.local_size(), undef);
    size_t                        level = 0;
    if (g.is_local(s)) {
        frontier.push_back(s);
    }
    while (!is_empty(frontier, comm)) {
        next_frontier = expand_frontier(g, level, frontier, dist);
        frontier      = exchange(std::move(next_frontier), comm);
        ++level;
    }
    return dist;
}
} // namespace bfs

template <typename T>
auto repeat_n(std::vector<T> const& vec, std::size_t n) {
    std::vector<T> result;
    for (size_t i = 0; i < n; ++i) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

TEST(ExamplesFromPaperTest, figure1) {
    using namespace kamping;
    Communicator        comm;
    std::vector<double> v            = {0.1, 3.14, 4.2, 123.4};
    std::vector<double> expected_res = repeat_n(v, comm.size());
    std::vector<int>    expected_rcounts(comm.size(), 4);
    std::vector<int>    expected_rdispls(comm.size());
    std::exclusive_scan(expected_rcounts.begin(), expected_rcounts.end(), expected_rdispls.begin(), 0);

    {
        // KaMPIng allows concise code
        // with sensible defaults ... (1)

        auto v_global = comm.allgatherv(send_buf(v));
        // test result (not part of listing)
        EXPECT_EQ(v_global, expected_res);
    }
    {
        // ... or detailed tuning of each parameter (2)
        std::vector<int> rc;
        auto [v_global, rcounts, rdispls] = comm.allgatherv(
            send_buf(v),                                           //(3)
            recv_counts_out<resize_to_fit /*(6)*/>(std::move(rc)), //(4)
            recv_displs_out()                                      //(5)
        );
        // test result (not part of listing)
        EXPECT_EQ(v_global, expected_res);
        EXPECT_EQ(rcounts, expected_rcounts);
        EXPECT_EQ(rdispls, expected_rdispls);
    }
}

TEST(ExamplesFromPaperTest, figure2) {
    using T                 = int;
    MPI_Datatype   MPI_TYPE = MPI_INT;
    MPI_Comm       comm     = MPI_COMM_WORLD;
    std::vector<T> v        = {1, 3, 4}; // fill with data
    int            size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    std::vector<int> rc(static_cast<size_t>(size)), rd(static_cast<size_t>(size));
    rc[static_cast<size_t>(rank)] = static_cast<int>(v.size());
    // exchange counts
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rc.data(), 1, MPI_INT, comm);
    // compute displacements
    std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
    int n_glob = rc.back() + rd.back();
    // allocate receive buffer
    std::vector<T> v_glob(static_cast<size_t>(n_glob));
    // exchange
    MPI_Allgatherv(v.data(), static_cast<int>(v.size()), MPI_TYPE, v_glob.data(), rc.data(), rd.data(), MPI_TYPE, comm);

    // test result (not part of listing)
    EXPECT_EQ(v_glob, repeat_n(v, static_cast<size_t>(size)));
}

TEST(ExamplesFromPaperTest, figure3) {
    using namespace kamping;
    Communicator comm;

    std::vector<int> v = {1, 3, 4}; // fill with data
    using T            = int;

    {
        // Version 1: using KaMPIng’s interface
        std::vector<int> rc(comm.size()), rd(comm.size());
        rc[comm.rank()] = static_cast<int>(v.size());
        comm.allgather(send_recv_buf(rc));
        std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
        std::vector<T> v_glob(static_cast<size_t>(rc.back() + rd.back()));
        comm.allgatherv(send_buf(v), recv_buf(v_glob), recv_counts(rc), recv_displs(rd));

        // test result (not part of listing)
        EXPECT_EQ(v_glob, repeat_n(v, comm.size()));
    }
    {
        // Version 2: displacements are computed implicitly
        std::vector<int> rc(comm.size());
        rc[comm.rank()] = static_cast<int>(v.size());
        comm.allgather(send_recv_buf(rc));
        std::vector<T> v_glob;
        comm.allgatherv(send_buf(v), recv_buf<resize_to_fit>(v_glob), recv_counts(rc));

        // test result (not part of listing)
        EXPECT_EQ(v_glob, repeat_n(v, comm.size()));
    }
    {
        // Version 3: counts are automatically exchanged
        // and result is returned by value
        std::vector<T> v_glob = comm.allgatherv(send_buf(v));

        // test result (not part of listing)
        EXPECT_EQ(v_glob, repeat_n(v, comm.size()));
    }
}

TEST(ExamplesFromPaperTest, section_III_snippets) {
    using namespace kamping;
    Communicator     comm;
    std::vector<int> v = {1, 3, 4}; // fill with data
    {
        auto result   = comm.allgatherv(send_buf(v), recv_counts_out());
        auto recv_buf = result.extract_recv_buf();
        auto counts   = result.extract_recv_counts();

        EXPECT_EQ(recv_buf, repeat_n(v, comm.size()));
        EXPECT_EQ(counts.size(), comm.size());
        EXPECT_THAT(counts, Each(3));
    }
    {
        auto [recv_buf, counts] = comm.allgatherv(send_buf(v), recv_counts_out());

        // test result (not part of listing)
        EXPECT_EQ(recv_buf, repeat_n(v, comm.size()));
        EXPECT_EQ(counts.size(), comm.size());
        EXPECT_THAT(counts, Each(3));
    }
    {
        using T = int;
        std::vector<T> tmp(comm.size() * v.size()); // ...
        // tmp is moved to the underlying call where the
        // storage is reused for the recv buffer
        auto recv_buffer = comm.allgatherv(send_buf(v), recv_buf(std::move(tmp)));

        // test result (not part of listing)
        EXPECT_EQ(recv_buffer, repeat_n(v, comm.size()));
    }
    {
        using T = int;
        std::vector<T> recv_buffer(comm.size() * v.size()); //...
        // data is written into recv_buffer directly
        comm.allgatherv(send_buf(v), recv_buf(recv_buffer));

        // test result (not part of listing)
        EXPECT_EQ(recv_buffer, repeat_n(v, comm.size()));
    }
    {
        using T = int;
        std::vector<T>   recv_buffer;         // has to be resized
        std::vector<int> counts(comm.size()); // size large enough
        comm.allgatherv(send_buf(v), recv_buf<resize_to_fit>(recv_buffer), recv_counts_out(counts));

        // test result (not part of listing)
        EXPECT_EQ(recv_buffer, repeat_n(v, comm.size()));
    }
}

TEST(ExamplesFromPaperTest, figure5) {
    using namespace kamping;
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }
    using dict = std::unordered_map<std::string, std::string>;
    dict data  = {{"foo", "bar"}, {"baz", "x"}};
    if (comm.rank() == 0) // if is not part of listing
        comm.send(send_buf(as_serialized(data)), destination(1));
    if (comm.rank() == 1) { // if is not part of listing
        dict recv_dict = comm.recv(recv_buf(as_deserializable<dict>()));
        // test result (not part of listing)
        EXPECT_EQ(recv_dict, data);
    }
}

TEST(ExamplesFromPaperTest, figure6) {
    using namespace kamping;
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }

    std::vector<int>       v          = {1, 3, 5}; // ...
    std::vector<int> const expected_v = {1, 3, 5}; // not part of listing
    if (comm.rank() == 0) {                        // if is not part of listing
        auto r1 = comm.isend(send_buf_out(std::move(v)), destination(1));
        v       = r1.wait(); // v is moved back to caller after
        // test result (not part of listing)
        EXPECT_EQ(v, expected_v);
    }
    if (comm.rank() == 1) { // if is not part of listing
        auto             r2   = comm.irecv<int>(recv_count(42));
        std::vector<int> data = r2.wait(); // data only returned
                                           // after request
                                           // is complete

        // test result (not part of listing)
        EXPECT_EQ(data.size(), 42);
        EXPECT_THAT(Span<int>(data.begin(), data.begin() + 3), ElementsAre(1, 3, 5));
    }
}

TEST(ExamplesFromPaperTest, section_III_g) {
    using namespace kamping;
    Communicator comm;

    std::vector<int> data(comm.size());
    data[comm.rank()] = static_cast<int>(comm.rank());
    data              = comm.allgather(send_recv_buf(std::move(data)));

    // test result (not part of listing)
    std::vector<int> expected_res(comm.size());
    std::iota(expected_res.begin(), expected_res.end(), 0);
    EXPECT_EQ(data, expected_res);
}

TEST(ExamplesFromPaperTest, figure7) {
    std::vector<int> data          = {13, 1, 7, 18};
    auto             gathered_data = kamping::comm_world().allgatherv(kamping::send_buf(data));
    std::sort(gathered_data.begin(), gathered_data.end());

    sorting::sort(data, MPI_COMM_WORLD);

    // test result (not part of listing)
    using namespace kamping;
    Communicator comm;
    auto         gathered_result = kamping::comm_world().allgatherv(kamping::send_buf(data));
    EXPECT_EQ(gathered_result, gathered_data);
}

TEST(ExamplesFromPaperTest, figure9) {
    using namespace kamping;
    Communicator comm;

    if (comm.size() != 4) {
        return;
    }

    // test result (not part of listing)
    bfs::Graph g          = bfs::init_graph();
    auto       bfs_levels = bfs::bfs(g, 0, MPI_COMM_WORLD);

    auto gathered_levels = comm.allgatherv(send_buf(bfs_levels));
    EXPECT_THAT(gathered_levels, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}
