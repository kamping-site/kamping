// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
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
#include <mdspan>
#include <vector>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/recv.hpp" // IWYU pragma: keep
#include "kamping/p2p/send.hpp" // IWYU pragma: keep
#include "kamping/v2/contrib/cereal_view.hpp"
#include "kamping/v2/views.hpp"

struct example_struct {
    int    foo;
    double bar;

    template <typename Archive>
    void serialize(Archive& ar) {
        ar(foo, bar);
    }
};

MPI_Datatype example_type() {
    int const    nitems          = 2;
    int          blocklengths[2] = {1, 1};
    MPI_Datatype types[2]        = {MPI_INT, MPI_DOUBLE};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(example_struct, foo);
    offsets[1] = offsetof(example_struct, bar);

    MPI_Datatype example_type;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &example_type);
    MPI_Type_commit(&example_type);
    return example_type;
}

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    KAMPING_ASSERT(comm.size() == 2uz, "This example must be run with exactly 2 ranks.");

    // {
    //     if (comm.rank_signed() == 0) {
    //         std::vector<int>                             data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    //         std::mdspan<int, std::extents<size_t, 3, 3>> to_send(data.data());
    //         comm.send(adapter::MDSpanAdapter(to_send), 1);
    //     } else {
    //         std::vector<int>                             data(9);
    //         std::mdspan<int, std::extents<size_t, 3, 3>> to_recv(data.data());
    //         auto                                         received = comm.recv(adapter::MDSpanAdapter(to_recv), 0);
    //         auto                                         result   = received.get_mdspan();
    //     }
    // }

    // {
    //     if (comm.rank_signed() == 0) {
    //         std::random_device rd;
    //         std::mt19937       gen(rd());

    //         std::uniform_int_distribution<> dist(1, 10);
    //         int                             ext1 = dist(gen);
    //         int                             ext2 = dist(gen);

    //         std::vector<int>                                                                 data(ext1 * ext2, 42);
    //         std::mdspan<int, std::extents<size_t, std::dynamic_extent, std::dynamic_extent>> to_send(
    //             data.data(),
    //             ext1,
    //             ext2
    //         );

    //         comm.send(std::ranges::single_view(ext1), 1);
    //         comm.send(std::ranges::single_view(ext2), 1);
    //         comm.send(adapter::MDSpanAdapter(to_send), 1);
    //     } else {
    //         std::ranges::single_view v1(0);
    //         comm.recv(v1, 0);

    //         std::ranges::single_view v2(0);
    //         comm.recv(v2, 0);

    //         int ext1 = v1.front();
    //         int ext2 = v2.front();

    //         std::vector<int>                                                                 data(ext1 * ext2);
    //         std::mdspan<int, std::extents<size_t, std::dynamic_extent, std::dynamic_extent>> to_recv(
    //             data.data(),
    //             ext1,
    //             ext2
    //         );

    //         auto received = comm.recv(adapter::MDSpanAdapter(to_recv), 0);

    //         auto result = received.get_mdspan();
    //     }
    // }

    {
        if (comm.rank_signed() == 0) {
            comm.send(std::vector{1, 2, 3, 4}, 1);

        } else {
            auto rbuf = comm.recv(std::vector<int>(4) | views::with_type(MPI_INT), 0);
            std::ranges::sort(rbuf);
            rbuf[0] = 3;
        }
    }
    // std::views::take: a standard library view that is not derived from view_interface_base.
    // kamping::ranges::size/data/type dispatch through the std::ranges CPOs via the
    // sized_range/contiguous_range fallback overloads — no kamping wrapping needed.
    {
        if (comm.rank_signed() == 0) {
            std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8};
            comm.send(data | std::views::take(4), 1); // sends only first 4 elements
        } else {
            std::vector<int> buf(4);
            comm.recv(buf | std::views::take(4), 0);
            KAMPING_ASSERT(buf[0] == 1 && buf[3] == 4, "take(4) recv mismatch.");
        }
    }

    // with_type on a vector: annotate a range with a custom MPI datatype
    {
        if (comm.rank_signed() == 0) {
            std::vector<example_struct> data(10, {42, 42.42});
            comm.send(data | kamping::views::with_type(example_type()), 1);
        } else {
            std::vector<example_struct> data(10);
            // auto                        rbuf = data | kamping::views::with_type(example_type());
            comm.recv(data | kamping::views::with_type(example_type()), 0);
        }
    }

    // with_type on a non-range: a plain struct that provides mpi_data/mpi_size but no mpi_type.
    // with_type adds the missing type annotation so the result satisfies data_buffer.
    {
        struct single_buffer {
            example_struct value;
            void*          mpi_data() {
                return &value;
            }
            std::size_t mpi_size() const {
                return 1;
            }
        };

        if (comm.rank_signed() == 0) {
            single_buffer buf{{42, 42.42}};
            auto          view = buf | kamping::views::with_type(example_type());
            static_assert(kamping::ranges::data_buffer<decltype(view)>);
            // comm.send(view, 1);
        }
    }

    // Pre-bound adaptor: store the partial application as a value and reuse it.
    {
        auto const typed_struct = kamping::views::with_type(example_type());

        std::vector<example_struct> a(5, {1, 1.0});
        std::vector<example_struct> b(5, {2, 2.0});
        (void)(a | typed_struct);
        (void)(b | typed_struct);
    }

    // Closure composition: a struct exposing only mpi_data() becomes a full data_buffer
    // by piping with_size and with_type. Neither view alone is sufficient.
    {
        struct data_only_buffer {
            example_struct value;
            void*          mpi_data() {
                return &value;
            }
        };

        auto const as_single_struct = kamping::views::with_size(1) | kamping::views::with_type(example_type());

        if (comm.rank_signed() == 0) {
            data_only_buffer buf{{42, 42.42}};
            auto             view = buf | as_single_struct;
            static_assert(kamping::ranges::data_buffer<decltype(view)>);
            // comm.send(view, 1);
        }
    }

    // views::resize: auto-sizing recv buffer.
    // The size is not known upfront; infer() probes MPI and calls set_recv_count(n),
    // which triggers the actual resize lazily on first mpi_data() access.
    {
        if (comm.rank_signed() == 0) {
            std::vector<int> data{10, 20, 30, 40, 50};
            comm.send(data, 1);
        } else {
            std::vector<int> buf; // starts empty — size inferred at recv time
            auto             rbuf = comm.recv(buf | kamping::views::resize, 0);
            std::println("Received {}.", rbuf);
            KAMPING_ASSERT(rbuf.base().base().size() == 5uz, "Expected 5 elements after auto-resize.");
        }
    }
    // views::serialize: send and receive non-contiguous / non-trivial objects via cereal.
    // Rank 0 serializes the map to bytes and sends. Rank 1 receives the bytes and
    // deserializes lazily when operator* is first called (triggering do_deserialize()).
    {
        if (comm.rank_signed() == 0) {
            std::unordered_map<std::string, int> map{{"a", 1}, {"b", 2}, {"c", 3}};
            comm.send(map | kamping::views::serialize, 1);
        } else {
            std::unordered_map<std::string, int> map;
            auto                                 view = comm.recv(map | kamping::views::serialize, 0);
            // operator* triggers deserialization; dereference before iterating.
            for (auto const& [k, v]: *view) {
                std::println("  {}: {}", k, v);
            }
            KAMPING_ASSERT(map.size() == 3uz, "Expected 3 entries after deserialization.");
        }
    }

    // views::serialize on a non-range type: operator* and operator-> trigger deserialization.
    {
        if (comm.rank_signed() == 0) {
            example_struct s{42, 3.14};
            comm.send(s | kamping::views::serialize, 1);
        } else {
            example_struct s{};
            auto           view = comm.recv(s | kamping::views::serialize, 0);
            std::println("foo={} bar={}", view->foo, view->bar);
            KAMPING_ASSERT((*view).foo == 42, "Deserialization mismatch.");
        }
    }

    // views::deserialize<T>: recv into an owning view without a pre-existing object.
    {
        if (comm.rank_signed() == 0) {
            example_struct s{7, 2.71};
            comm.send(s | kamping::views::serialize, 1);
        } else {
            auto view = comm.recv(kamping::views::deserialize<example_struct>(), 0);
            std::println("foo={} bar={}", view->foo, view->bar);
            KAMPING_ASSERT((*view).foo == 7, "deserialize<T> mismatch.");
        }
    }
}
