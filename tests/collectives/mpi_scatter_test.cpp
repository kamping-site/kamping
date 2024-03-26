// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "../test_assertions.hpp"

#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

namespace {
template <template <typename...> typename DefaultContainerType>
std::vector<int> create_input_vector_on_root(
    Communicator<DefaultContainerType> const& comm, int const elements_per_rank, int root = -1
) {
    if (root < 0) {
        root = comm.root_signed();
    }

    std::vector<int> input;
    if (comm.rank_signed() == root) {
        input.resize(static_cast<std::size_t>(elements_per_rank) * comm.size());
        for (int rank = 0; rank < comm.size_signed(); ++rank) {
            auto begin = input.begin() + rank * elements_per_rank;
            auto end   = begin + elements_per_rank;
            std::fill(begin, end, rank);
        }
    }
    return input;
}
} // namespace

TEST(ScatterTest, scatter_single_element_no_recv_buffer) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_recv_buffer) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_explicit_send_count_and_recv_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size() * 2, 0);
    std::iota(input.begin(), input.begin() + comm.size_signed(), 0);
    std::vector<int> result;

    // test that send_count parameter overwrites automatic deduction of send counts from the size of the send buffer
    comm.scatter(send_buf(input), send_count(1), recv_buf<resize_to_fit>(result));
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_explicit_send_count_only_at_root) {
    Communicator comm;

    std::vector<int> input(comm.size() * 2, 0);
    std::iota(input.begin(), input.begin() + comm.size_signed(), 0);
    std::vector<int> result;

    int const root = comm.size_signed() / 2;
    // test that send_count parameter overwrites automatic deduction of send counts from the size of the send buffer
    // even if this value is only given at the root.
    if (comm.is_root(root)) {
        comm.scatter(send_buf(input), send_count(1), kamping::root(root), recv_buf<resize_to_fit>(result));
    } else {
        comm.scatter(send_buf(input), kamping::root(root), recv_buf<resize_to_fit>(result));
    }
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter_single(send_buf(input));
    EXPECT_EQ(result, comm.rank());
}

TEST(ScatterTest, scatter_single_with_owning_send_buf) {
    Communicator comm;

    auto       input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter_single(send_buf(std::move(input)));
    EXPECT_EQ(result, comm.rank());
}

TEST(ScatterTest, scatter_single_with_explicit_root) {
    Communicator comm;

    int const  root   = comm.size_signed() - 1;
    auto const input  = create_input_vector_on_root(comm, 1, root);
    auto const result = comm.scatter_single(send_buf(input), kamping::root(root));
    EXPECT_EQ(result, comm.rank());
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST(ScatterTest, scatter_single_with_too_small_send_buf) {
    Communicator comm;

    std::vector<int> const input = create_input_vector_on_root(comm, 1);
    if (comm.is_root()) {
        std::vector<int> const input_too_small(input.begin(), std::next(input.begin(), comm.size_signed() / 2));
        EXPECT_KASSERT_FAILS(
            comm.scatter_single(send_buf(input_too_small)),
            "send_buf of size equal to comm.size() must be provided on the root rank."
        );
        // scatter call to catch other ranks waiting on the above failed scatter-call
        comm.scatter_single(send_buf(input));
    } else {
        comm.scatter_single(send_buf(input));
    }
}

TEST(ScatterTest, scatter_single_with_too_small_send_buf_and_explicit_root) {
    Communicator comm;

    int const              root  = comm.size_signed() - 1;
    std::vector<int> const input = create_input_vector_on_root(comm, 1, root);
    if (comm.is_root(root)) {
        std::vector<int> const input_too_small(input.begin(), std::next(input.begin(), comm.size_signed() / 2));
        EXPECT_KASSERT_FAILS(
            comm.scatter_single(send_buf(input_too_small), kamping::root(root)),
            "send_buf of size equal to comm.size() must be provided on the root rank."
        );
        // scatter call to catch other ranks waiting on the above failed scatter-call
        comm.scatter_single(send_buf(input), kamping::root(root));
    } else {
        comm.scatter_single(send_buf(input), kamping::root(root));
    }
}
#endif

TEST(ScatterTest, scatter_send_count_parameter_is_only_considered_at_root) {
    Communicator comm;

    int const        root = comm.size_signed() / 2;
    std::vector<int> input(comm.size());
    if (comm.is_root(root)) {
        std::iota(input.begin(), input.end(), 0);
    } else {
        input.resize(29);
    }
    std::vector<int> result;

    // test that auto deduction for send count happens only at the root processor.
    if (comm.is_root(root)) {
        comm.scatter(send_buf(input), kamping::root(root), recv_buf<resize_to_fit>(result));
    } else {
        comm.scatter(send_buf(input), kamping::root(root), recv_buf<resize_to_fit>(result));
    }
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_recv_count) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input), recv_count(1));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_extract_recv_count) {
    Communicator comm;

    auto const input = create_input_vector_on_root(comm, 1);

    EXPECT_EQ(comm.scatter(send_buf(input), recv_count_out()).extract_recv_count(), 1);

    int recv_count_value;
    comm.scatter(send_buf(input), recv_count_out(recv_count_value));
    EXPECT_EQ(recv_count_value, 1);
}

TEST(ScatterTest, scatter_multiple_elements) {
    int const elements_per_pe = 4;

    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, elements_per_pe);
    auto const result = comm.scatter(send_buf(input));

    ASSERT_EQ(result.size(), elements_per_pe);
    EXPECT_THAT(result, Each(comm.rank()));
}

TEST(ScatterTest, scatter_with_send_buf_ignore_with_recv_buf) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    if (comm.is_root()) {
        comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result));
    } else {
        comm.scatter(send_buf(ignore<int>), recv_buf<resize_to_fit>(result));
    }

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_send_buf_ignore) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = (comm.is_root()) ? comm.scatter(send_buf(input)) : comm.scatter(send_buf(ignore<int>));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_send_buf_only_on_root_with_recv_buf) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    if (comm.is_root()) {
        comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result));
    } else {
        comm.scatter(recv_buf<resize_to_fit>(result));
    }

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_send_buf_only_on_root) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = (comm.is_root()) ? comm.scatter(send_buf(input)) : comm.scatter<int>();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_root_arg) {
    Communicator comm;
    int const    root = comm.size_signed() - 1; // use last PE as root

    auto const input  = create_input_vector_on_root(comm, 1, root);
    auto const result = comm.scatter(send_buf(input), kamping::root(root));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_nonzero_root_comm) {
    Communicator comm;
    comm.root(comm.size() - 1);

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_recv_count_out) {
    Communicator comm;

    auto const input = create_input_vector_on_root(comm, 2);
    int        recv_count;
    auto const result = comm.scatter(send_buf(input), recv_count_out(recv_count));

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(recv_count, 2);
}

TEST(ScatterTest, scatter_with_custom_sendbuf_and_type) {
    Communicator comm;
    struct Data {
        int value;
    };

    ::testing::OwnContainer<Data> input(static_cast<std::size_t>(comm.size()));
    if (comm.is_root()) {
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            input[rank].value = asserting_cast<int>(rank);
        }
    }

    auto const result = comm.scatter(send_buf(input));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front().value, comm.rank());
}

TEST(ScatterTest, scatter_with_nonempty_sendbuf_on_non_root) {
    Communicator comm;

    std::vector<int> input(static_cast<std::size_t>(comm.size()));
    for (size_t rank = 0; rank < comm.size(); ++rank) {
        input[rank] = asserting_cast<int>(rank);
    }

    auto const result = comm.scatter(send_buf(input));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT_COMMUNICATION)
TEST(ScatterTest, scatter_different_roots_on_different_processes) {
    Communicator comm;
    auto const   input = create_input_vector_on_root(comm, 1);
    if (comm.size() > 1) {
        EXPECT_KASSERT_FAILS(comm.scatter(send_buf(input), root(comm.rank())), "");
    }
}
#endif

TEST(ScatterTest, scatter_default_container_type) {
    Communicator<OwnContainer> comm;
    std::vector<int> const     input = create_input_vector_on_root(comm, 1);

    // This just has to compile
    OwnContainer<int> const result = comm.scatter(send_buf(input));
}

TEST(ScatterTest, scatter_single_element_with_given_recv_buf_bigger_than_required) {
    Communicator     comm;
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    {
        // recv buffer will be resized as resize policy is resize_to_fit
        std::vector<int> result{0, -1};
        comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> result{0, -1};
        comm.scatter(send_buf(input), recv_buf<grow_only>(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
    {
        // recv buffer will not be resized as policy is no_resize
        std::vector<int> result{0, -1};
        comm.scatter(send_buf(input), recv_buf<no_resize>(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
    {
        // recv buffer will not be resized as default policy is no_resize
        std::vector<int> result{0, -1};
        comm.scatter(send_buf(input), recv_buf(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
}

TEST(ScatterTest, scatter_single_element_with_given_recv_buf_smaller_than_required) {
    Communicator     comm;
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    {
        // recv buffer will be resized as resize policy is resize_to_fit
        std::vector<int> result;
        comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
    {
        // recv buffer will be resized as resize policy is grow_only
        std::vector<int> result;
        comm.scatter(send_buf(input), recv_buf<grow_only>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    {
        // recv buffer will not be resized as policy is no_resize; therefore the kassert for a sufficiently sized recv
        // buffer will fail
        std::vector<int> result;
        EXPECT_KASSERT_FAILS(comm.scatter(send_buf(input), recv_buf<no_resize>(result)), "");
    }

    {
        // recv buffer will not be resized as default policy is no_resize; therefore the kassert for a sufficiently
        // sized recv buffer will fail
        std::vector<int> result;
        EXPECT_KASSERT_FAILS(comm.scatter(send_buf(input), recv_buf(result)), "");
    }
#endif
}

TEST(ScatterTest, scatter_send_recv_count_are_out_parameters) {
    Communicator comm;

    auto const       input      = create_input_vector_on_root(comm, 1);
    int              send_count = -1;
    int              recv_count = -1;
    std::vector<int> result;
    comm.scatter(
        send_buf(input),
        recv_buf<resize_to_fit>(result),
        send_count_out(send_count),
        recv_count_out(recv_count)
    );

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 1);
    } else {
        // do not touch send_count on non-root parameters
        send_count = -1;
    }
    EXPECT_EQ(recv_count, 1);
}

TEST(ScatterTest, scatter_send_recv_count_are_part_of_result_object) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    auto res = comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result), send_count_out(), recv_count_out());

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    if (comm.is_root()) {
        EXPECT_EQ(res.extract_send_count(), 1);
    }
    EXPECT_EQ(res.extract_recv_count(), 1);
}

TEST(ScatterTest, scatter_send_recv_type_are_out_parameters) {
    Communicator comm;

    auto const       input     = create_input_vector_on_root(comm, 1);
    MPI_Datatype     send_type = MPI_CHAR;
    MPI_Datatype     recv_type = MPI_CHAR;
    std::vector<int> result;
    comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result), send_type_out(send_type), recv_type_out(recv_type));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    EXPECT_EQ(send_type, MPI_INT);
    EXPECT_EQ(recv_type, MPI_INT);
}

TEST(ScatterTest, scatter_send_recv_type_are_part_of_result_object) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    auto res = comm.scatter(send_buf(input), recv_buf<resize_to_fit>(result), send_type_out(), recv_type_out());

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    EXPECT_EQ(res.extract_send_type(), MPI_INT);
    EXPECT_EQ(res.extract_recv_type(), MPI_INT);
}

TEST(ScatterTest, non_trivial_send_type) {
    // root rank sends sends each rank its rank two times with padding and all ranks receive the messages without
    // padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(6 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            input[6 * i]     = static_cast<int>(i);
            input[6 * i + 3] = static_cast<int>(i);
        }
    }
    std::vector<int> recv_buffer(2, 0);

    MPI_Type_commit(&int_padding_padding);
    auto res = comm.scatter(
        send_buf(input),
        send_type(int_padding_padding),
        send_count(2),
        recv_buf(recv_buffer),
        recv_count_out()
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_EQ(res.extract_recv_count(), 2);
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed(), comm.rank_signed()));
}

TEST(ScatterTest, non_trivial_recv_type) {
    // root rank sends sends each rank its rank two times and all ranks receive the messages with padding.
    Communicator comm;
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();

    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(2 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            input[2 * i]     = static_cast<int>(i);
            input[2 * i + 1] = static_cast<int>(i);
        }
    }

    int const        init_value = -1;
    std::vector<int> recv_buffer(6, init_value);
    int              send_count = -1;

    MPI_Type_commit(&int_padding_padding);
    comm.scatter(
        send_buf(input),
        send_count_out(send_count),
        recv_buf(recv_buffer),
        recv_type(int_padding_padding),
        recv_count(2)
    );
    MPI_Type_free(&int_padding_padding);

    if (comm.is_root()) {
        EXPECT_EQ(send_count, 2);
    } else {
        // do not touch send_count on non-root ranks
        EXPECT_EQ(send_count, -1);
    }
    EXPECT_THAT(
        recv_buffer,
        ElementsAre(comm.rank_signed(), init_value, init_value, comm.rank_signed(), init_value, init_value)
    );
}
TEST(ScatterTest, different_send_and_recv_counts) {
    // root rank sends sends each rank its rank two times and all ranks receive the two messages at once.
    Communicator     comm;
    MPI_Datatype     int_padding_int = MPI_INT_padding_MPI_INT();
    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(2 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            input[2 * i]     = static_cast<int>(i);
            input[2 * i + 1] = static_cast<int>(i);
        }
    }

    int const        init_value = -1;
    std::vector<int> recv_buffer(3, init_value);
    int              send_count = -1;

    MPI_Type_commit(&int_padding_int);
    comm.scatter(
        send_buf(input),
        send_count_out(send_count),
        recv_buf(recv_buffer),
        recv_type(int_padding_int),
        recv_count(1)
    );
    MPI_Type_free(&int_padding_int);

    if (comm.is_root()) {
        EXPECT_EQ(send_count, 2);
    } else {
        // do not touch send_count on non-root ranks
        EXPECT_EQ(send_count, -1);
    }
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed(), init_value, comm.rank_signed()));
}
struct CustomRecvStruct {
    int  a;
    int  b;
    bool operator==(CustomRecvStruct const& other) const {
        return std::tie(a, b) == std::tie(other.a, other.b);
    }
    friend std::ostream& operator<<(std::ostream& out, CustomRecvStruct const& str) {
        return out << "(" << str.a << ", " << str.b << ")";
    }
};

TEST(ScatterTest, different_send_and_recv_counts_without_explicit_mpi_types) {
    Communicator comm;

    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(2 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            input[2 * i]     = static_cast<int>(i);
            input[2 * i + 1] = static_cast<int>(i);
        }
    }
    std::vector<CustomRecvStruct> recv_buffer(1);
    int                           send_count = -1;

    comm.scatter(send_buf(input), send_count_out(send_count), recv_count(1), recv_buf(recv_buffer));

    CustomRecvStruct expected_result{comm.rank_signed(), comm.rank_signed()};
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 2);
    } else {
        // do not touch send_count on non-root ranks
        EXPECT_EQ(send_count, -1);
    }
    EXPECT_THAT(recv_buffer, ElementsAre(expected_result));
}

TEST(ScatterTest, structured_bindings_explicit_recv_buf) {
    Communicator comm;
    auto const   input = create_input_vector_on_root(comm, 1);

    std::vector<int> recv_buffer(1);
    auto [recv_count, send_count, recv_type, send_type] = comm.scatter(
        send_buf(input),
        recv_count_out(),
        recv_buf(recv_buffer),
        send_count_out(),
        recv_type_out(),
        send_type_out()
    );

    EXPECT_EQ(recv_type, MPI_INT);
    EXPECT_EQ(recv_count, 1);
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(send_type, MPI_INT);
    } else {
        EXPECT_EQ(send_count, 0);
    }
}

TEST(ScatterTest, structured_bindings_implicit_recv_buf) {
    Communicator comm;
    auto const   input = create_input_vector_on_root(comm, 1);
    auto [recv_buffer, recv_count, send_count, recv_type, send_type] =
        comm.scatter(send_buf(input), recv_count_out(), send_count_out(), recv_type_out(), send_type_out());

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(send_type, MPI_INT);
    } else {
        EXPECT_EQ(send_count, 0);
    }
}

TEST(ScatterTest, structured_bindings_explicit_owning_recv_buf) {
    Communicator comm;
    auto const   input                                               = create_input_vector_on_root(comm, 1);
    auto [recv_count, send_count, recv_type, send_type, recv_buffer] = comm.scatter(
        send_buf(input),
        recv_count_out(),
        send_count_out(),
        recv_type_out(),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{})
    );

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_type, MPI_INT);
        EXPECT_EQ(send_count, 1);
    } else {
        EXPECT_EQ(send_count, 0);
    }
}

TEST(ScatterTest, structured_bindings_explicit_owning_recv_buf_non_owning_send_count) {
    Communicator comm;
    auto const   input                                   = create_input_vector_on_root(comm, 1);
    int          send_count                              = -1;
    auto [recv_count, recv_type, send_type, recv_buffer] = comm.scatter(
        send_buf(input),
        recv_count_out(),
        send_count_out(send_count),
        recv_type_out(),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{})
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(send_type, MPI_INT);
    } else {
        EXPECT_EQ(send_count, -1);
    }
}
TEST(ScatterTest, structured_bindings_explicit_owning_recv_buf_non_owning_send_count_recv_type) {
    Communicator comm;
    auto const   input      = create_input_vector_on_root(comm, 1);
    int          send_count = -1;
    MPI_Datatype recv_type;
    auto [recv_count, send_type, recv_buffer] = comm.scatter(
        send_buf(input),
        recv_count_out(),
        send_count_out(send_count),
        recv_type_out(recv_type),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{})
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(send_type, MPI_INT);
    } else {
        EXPECT_EQ(send_count, -1);
    }
}

TEST(
    ScatterTest,
    structured_bindings_explicit_owning_recv_buf_non_owning_send_count_recv_type_with_changed_order_and_root_param
) {
    Communicator comm;
    int const    root       = comm.size_signed() - 1;
    auto const   input      = create_input_vector_on_root(comm, 1, root);
    int          send_count = -1;
    MPI_Datatype recv_type;
    auto [recv_count, send_type, recv_buffer] = comm.scatter(
        send_count_out(send_count),
        recv_type_out(recv_type),
        recv_count_out(),
        send_buf(input),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{}),
        kamping::root(root)
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root(root)) {
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(send_type, MPI_INT);
    } else {
        EXPECT_EQ(send_count, -1);
    }
}
