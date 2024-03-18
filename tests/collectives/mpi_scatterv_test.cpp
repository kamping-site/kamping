// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
#include "kamping/span.hpp"

using namespace ::kamping;
using namespace ::testing;

namespace {
template <template <typename...> typename DefaultContainerType>
std::vector<int> create_equiv_sized_input_vector_on_root(
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

template <template <typename...> typename DefaultContainerType>
std::vector<int> create_equiv_counts_on_root(
    Communicator<DefaultContainerType> const& comm, int const elements_per_rank, int root = -1
) {
    if (root < 0) {
        root = comm.root_signed();
    }

    std::vector<int> counts;
    if (comm.rank_signed() == root) {
        counts.resize(comm.size());
        std::fill(counts.begin(), counts.end(), elements_per_rank);
    }
    return counts;
}
} // namespace

TEST(ScattervTest, scatterv_equiv_single_element_return_recv_buf) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    auto const result = comm.scatterv(send_buf(input), send_counts(counts), recv_count(1));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_buf) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    std::vector<int> result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_count(1), recv_buf<resize_to_fit>(result));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_buf_var) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    int        result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_count(1), recv_buf(result));

    EXPECT_EQ(result, comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_no_recv_count) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    auto const result = comm.scatterv(send_buf(input), send_counts(counts));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_count) {
    Communicator comm;

    auto const input      = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts     = create_equiv_counts_on_root(comm, 1);
    int        recv_count = -1;
    int        result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_count_out(recv_count), recv_buf<no_resize>(result));

    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_return_send_displs) {
    Communicator comm;

    auto const input       = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const send_counts = create_equiv_counts_on_root(comm, 1);
    int const  recv_count  = 1;
    int        result;
    auto       send_displs = comm.scatterv(
                               send_buf(input),
                               kamping::recv_count(recv_count),
                               kamping::send_counts(send_counts),
                               recv_buf(result),
                               send_displs_out()
    )
                           .extract_send_displs();

    if (comm.is_root()) {
        EXPECT_EQ(send_displs.size(), comm.size());
        for (std::size_t pe = 0; pe < comm.size(); ++pe) {
            EXPECT_EQ(send_displs[pe], pe);
        }
    }
    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_send_displs) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    int              result;
    std::vector<int> displs;
    comm.scatterv(send_buf(input), send_counts(counts), send_displs_out<resize_to_fit>(displs), recv_buf(result));

    if (comm.is_root()) {
        EXPECT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe);
        }
    } else {
        // displacement buffer should not be touched on non-root PEs
        EXPECT_EQ(displs.size(), 0);
    }

    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_multiple_elements) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, comm.size_signed());
    auto const       counts = create_equiv_counts_on_root(comm, comm.size_signed());
    std::vector<int> displs;
    int              recv_count;
    auto const       result = comm.scatterv(
        send_buf(input),
        send_counts(counts),
        send_displs_out<resize_to_fit>(displs),
        recv_count_out(recv_count)
    );

    if (comm.is_root()) {
        ASSERT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe * comm.size_signed());
        }
    } else {
        // displacement buffer should not be touched on non-root PEs
        EXPECT_EQ(displs.size(), 0);
    }

    EXPECT_EQ(recv_count, comm.size_signed());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_equiv_multiple_elements_send_buf_only_on_root_no_receive_buf) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, comm.size_signed());
    auto const       counts = create_equiv_counts_on_root(comm, comm.size_signed());
    std::vector<int> displs;
    int              recv_count;
    std::vector<int> result;
    if (comm.is_root()) {
        result = comm.scatterv(
            send_buf(input),
            send_counts(counts),
            send_displs_out<resize_to_fit>(displs),
            recv_count_out(recv_count)
        );
    } else {
        result =
            comm.scatterv<int>(send_counts(counts), send_displs_out<resize_to_fit>(displs), recv_count_out(recv_count));
    }

    if (comm.is_root()) {
        ASSERT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe * comm.size_signed());
        }
    } else {
        // displacement buffer should not be touched on non-root PEs
        EXPECT_EQ(displs.size(), 0);
    }

    EXPECT_EQ(recv_count, comm.size_signed());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_equiv_multiple_elements_send_buf_only_on_root_with_receive_buf) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, comm.size_signed());
    auto const       counts = create_equiv_counts_on_root(comm, comm.size_signed());
    std::vector<int> displs;
    int              recv_count;
    std::vector<int> result;
    if (comm.is_root()) {
        result = comm.scatterv(
            send_buf(input),
            send_counts(counts),
            send_displs_out<resize_to_fit>(displs),
            recv_count_out(recv_count)
        );
    } else {
        comm.scatterv(
            recv_buf<resize_to_fit>(result),
            send_counts(counts),
            send_displs_out(displs),
            recv_count_out(recv_count)
        );
    }

    if (comm.is_root()) {
        ASSERT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe * comm.size_signed());
        }
    } else {
        // displacement buffer should not be touched on non-root PEs
        EXPECT_EQ(displs.size(), 0);
    }

    EXPECT_EQ(recv_count, comm.size_signed());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_nonequiv) {
    Communicator comm;

    // Send i+1 elements to PE i
    std::vector<int> input;
    std::vector<int> counts;
    for (int pe = 0; pe < comm.size_signed(); ++pe) {
        for (int i = 0; i <= pe; ++i) {
            input.push_back(pe);
        }
        counts.push_back(pe + 1);
    }

    int        recv_count;
    auto const result = comm.scatterv(send_buf(input), send_counts(counts), recv_count_out(recv_count));

    EXPECT_EQ(recv_count, comm.rank_signed() + 1);
    EXPECT_EQ(result.size(), comm.rank() + 1);
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_nonzero_root) {
    Communicator comm;
    int const    root_val = comm.size_signed() - 1;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1, root_val);
    auto const counts = create_equiv_counts_on_root(comm, 1, root_val);

    auto const result = comm.scatterv(send_buf(input), root(root_val), send_counts(counts), recv_count(1));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank_signed());
}

TEST(ScattervTest, scatterv_default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    std::vector<int> const counts = create_equiv_counts_on_root(comm, 1);
    auto                   result = comm.scatterv(
        send_buf(input),
        send_counts(counts),
        recv_count(1),
        send_displs_out(alloc_new_using<OwnContainer>)
    );

    // This just has to compile
    OwnContainer<int> recv_buf    = result.extract_recv_buffer();
    OwnContainer<int> send_displs = result.extract_send_displs();
}

TEST(ScattervTest, scatterv_single_element_with_given_recv_buf_bigger_than_required) {
    Communicator comm;
    auto const   input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const   counts = create_equiv_counts_on_root(comm, 1);
    {
        // recv buffer will be resized as resize policy is resize_to_fit
        std::vector<int> result{0, -1};
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf<resize_to_fit>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> result{0, -1};
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf<grow_only>(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
    {
        // recv buffer will not be resized as policy is no_resize
        std::vector<int> result{0, -1};
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf<no_resize>(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
    {
        // recv buffer will not be resized as default policy is no_resize
        std::vector<int> result{0, -1};
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf(result));
        ASSERT_EQ(result.size(), 2);
        EXPECT_THAT(result, ElementsAre(comm.rank(), -1));
    }
}

TEST(ScatterTest, scatterv_single_element_with_given_recv_buf_smaller_than_required) {
    Communicator comm;
    auto const   input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const   counts = create_equiv_counts_on_root(comm, 1);

    {
        // recv buffer will be resized as resize policy is resize_to_fit
        std::vector<int> result;
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf<resize_to_fit>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
    {
        // recv buffer will be resized as resize policy is grow_only
        std::vector<int> result;
        comm.scatterv(send_buf(input), send_counts(counts), recv_buf<grow_only>(result));
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result.front(), comm.rank());
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    {
        // recv buffer will not be resized as policy is no_resize; therefore the kassert for a sufficiently sized recv
        // buffer will fail
        std::vector<int> result;
        EXPECT_KASSERT_FAILS(comm.scatterv(send_buf(input), send_counts(counts), recv_buf<no_resize>(result)), "");
    }

    {
        // recv buffer will not be resized as default policy is no_resize; therefore the kassert for a sufficiently
        // sized recv buffer will fail
        std::vector<int> result;
        EXPECT_KASSERT_FAILS(comm.scatterv(send_buf(input), send_counts(counts), recv_buf(result)), "");
    }
#endif
}

TEST(ScattervTest, scatterv_single_element_with_given_send_displs_bigger_than_required) {
    Communicator     comm;
    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    std::vector<int> expected_send_displs_on_root(comm.size());
    std::exclusive_scan(counts.begin(), counts.end(), expected_send_displs_on_root.begin(), 0);
    int const default_value = 42;

    {
        // send displs buffer will be resized on root as resize policy is resize_to_fit
        std::vector<int> send_displs(2 * comm.size(), default_value);
        auto             recv_buf =
            comm.scatterv(send_buf(input), send_displs_out<resize_to_fit>(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), comm.size());
            EXPECT_EQ(send_displs, expected_send_displs_on_root);
        } else {
            // send displacements should not be altered on non-root PEs
            EXPECT_EQ(send_displs.size(), 2 * comm.size());
            EXPECT_EQ(send_displs, std::vector<int>(2 * comm.size(), default_value));
        }
    }
    {
        // send displs buffer will not be resized on root as it is large enough and resize policy is grow only
        std::vector<int> send_displs(2 * comm.size(), default_value);
        auto recv_buf = comm.scatterv(send_buf(input), send_displs_out<grow_only>(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), 2 * comm.size());
            // first half send displacements
            EXPECT_THAT(Span<int>(send_displs.data(), comm.size()), ElementsAreArray(expected_send_displs_on_root));
            // second half send displacements (should not be altered)
            EXPECT_THAT(
                Span<int>(send_displs.data() + comm.size(), comm.size()),
                ElementsAreArray(std::vector<int>(comm.size(), default_value))
            );
        } else {
            // send displacement should not be altered on non-root PEs
            EXPECT_EQ(send_displs, std::vector<int>(2 * comm.size(), default_value));
        }
    }
    {
        // send displs buffer will not be resized on root as resize policy is no_resize
        std::vector<int> send_displs(2 * comm.size(), default_value);
        auto recv_buf = comm.scatterv(send_buf(input), send_displs_out<no_resize>(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), 2 * comm.size());
            // first half send displacements
            EXPECT_THAT(Span<int>(send_displs.data(), comm.size()), ElementsAreArray(expected_send_displs_on_root));
            // second half send displacements (should not be altered)
            EXPECT_THAT(
                Span<int>(send_displs.data() + comm.size(), comm.size()),
                ElementsAreArray(std::vector<int>(comm.size(), default_value))
            );
        } else {
            // send displacement should not be altered on non-root PEs
            EXPECT_EQ(send_displs, std::vector<int>(2 * comm.size(), default_value));
        }
    }
    {
        // send displs buffer will not be resized on root as resize policy is no_resize (default value)
        std::vector<int> send_displs(2 * comm.size(), default_value);
        auto             recv_buf = comm.scatterv(send_buf(input), send_displs_out(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), 2 * comm.size());
            // first half send displacements
            EXPECT_THAT(Span<int>(send_displs.data(), comm.size()), ElementsAreArray(expected_send_displs_on_root));
            // second half send displacements (should not be altered)
            EXPECT_THAT(
                Span<int>(send_displs.data() + comm.size(), comm.size()),
                ElementsAreArray(std::vector<int>(comm.size(), default_value))
            );
        } else {
            // send displacement should not be altered on non-root PEs
            EXPECT_EQ(send_displs, std::vector<int>(2 * comm.size(), default_value));
        }
    }
}

TEST(ScattervTest, scatterv_single_element_with_given_send_displs_smaller_than_required) {
    Communicator     comm;
    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    std::vector<int> expected_send_displs_on_root(comm.size());
    std::exclusive_scan(counts.begin(), counts.end(), expected_send_displs_on_root.begin(), 0);

    {
        // send displs buffer will be resized on root as resize policy is resize_to_fit
        std::vector<int> send_displs;
        auto             recv_buf =
            comm.scatterv(send_buf(input), send_displs_out<resize_to_fit>(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), comm.size());
            EXPECT_EQ(send_displs, expected_send_displs_on_root);
        } else {
            // send displacements should not be altered on non-root PEs
            EXPECT_EQ(send_displs.size(), 0);
        }
    }
    {
        // send displs buffer will not be resized on root as it is large enough and resize policy is grow only
        std::vector<int> send_displs;
        auto recv_buf = comm.scatterv(send_buf(input), send_displs_out<grow_only>(send_displs), send_counts(counts));
        ASSERT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.rank());
        if (comm.is_root()) {
            EXPECT_EQ(send_displs.size(), comm.size());
            EXPECT_EQ(send_displs, expected_send_displs_on_root);
        } else {
            EXPECT_EQ(send_displs.size(), 0);
        }
    }
    // cannot test kassert for no_resize policy as this will lead to undefined MPI behaviour due to communication
    // attempts from non-root ranks when the kassert on the root has already failed.
}

TEST(ScattervTest, scatter_send_recv_type_are_out_parameters) {
    Communicator comm;

    auto const       input       = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       send_counts = create_equiv_counts_on_root(comm, 1);
    MPI_Datatype     send_type   = MPI_CHAR;
    MPI_Datatype     recv_type   = MPI_CHAR;
    std::vector<int> result;
    comm.scatterv(
        send_buf(input),
        recv_buf<resize_to_fit>(result),
        kamping::send_counts(send_counts),
        send_type_out(send_type),
        recv_type_out(recv_type)
    );

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    EXPECT_EQ(send_type, MPI_INT);
    EXPECT_EQ(recv_type, MPI_INT);
}

TEST(ScattervTest, scatter_send_recv_type_are_part_of_result_object) {
    Communicator comm;

    auto const       input       = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       send_counts = create_equiv_counts_on_root(comm, 1);
    std::vector<int> result;
    auto             res = comm.scatterv(
        send_buf(input),
        recv_buf<resize_to_fit>(result),
        kamping::send_counts(send_counts),
        send_type_out(),
        recv_type_out()
    );

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
    EXPECT_EQ(res.extract_send_type(), MPI_INT);
    EXPECT_EQ(res.extract_recv_type(), MPI_INT);
}

TEST(ScattervTest, non_trivial_send_type) {
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
    auto const       send_counts_buf = create_equiv_counts_on_root(comm, 2);
    std::vector<int> recv_buffer(2, 0);

    MPI_Type_commit(&int_padding_padding);
    auto res = comm.scatterv(
        send_buf(input),
        send_type(int_padding_padding),
        send_counts(send_counts_buf),
        recv_buf(recv_buffer),
        recv_count_out()
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_EQ(res.extract_recv_count(), 2);
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed(), comm.rank_signed()));
}

TEST(ScattervTest, non_trivial_recv_type) {
    // root rank sends sends each rank its rank two times and all ranks receive the messages with padding.
    Communicator comm;
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    auto const   send_counts_buf     = create_equiv_counts_on_root(comm, 2);

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

    MPI_Type_commit(&int_padding_padding);
    comm.scatterv(
        send_buf(input),
        send_counts(send_counts_buf),
        recv_buf(recv_buffer),
        recv_type(int_padding_padding),
        recv_count(2)
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_THAT(
        recv_buffer,
        ElementsAre(comm.rank_signed(), init_value, init_value, comm.rank_signed(), init_value, init_value)
    );
}

TEST(ScattervTest, different_send_and_recv_counts) {
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
    auto const       send_counts_buf = create_equiv_counts_on_root(comm, 2);

    MPI_Type_commit(&int_padding_int);
    comm.scatterv(
        send_buf(input),
        send_counts(send_counts_buf),
        recv_buf(recv_buffer),
        recv_type(int_padding_int),
        recv_count(1)
    );
    MPI_Type_free(&int_padding_int);

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed(), init_value, comm.rank_signed()));
}

TEST(ScattervTest, different_send_and_recv_counts_without_explicit_mpi_types) {
    Communicator comm;

    struct CustomRecvStruct {
        int  a;
        int  b;
        bool operator==(CustomRecvStruct const& other) const {
            return std::tie(a, b) == std::tie(other.a, other.b);
        }
    };
    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(2 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            input[2 * i]     = static_cast<int>(i);
            input[2 * i + 1] = static_cast<int>(i);
        }
    }
    std::vector<CustomRecvStruct> recv_buffer(1);
    auto const                    send_counts_buf = create_equiv_counts_on_root(comm, 2);

    comm.scatterv(send_buf(input), send_counts(send_counts_buf), recv_count(1), recv_buf(recv_buffer));

    CustomRecvStruct expected_result{comm.rank_signed(), comm.rank_signed()};
    EXPECT_THAT(recv_buffer, ElementsAre(expected_result));
}

TEST(ScattervTest, structured_bindings_explicit_recv_buf) {
    Communicator     comm;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1);
    std::vector<int> send_counts;
    if (comm.is_root()) {
        send_counts = std::vector<int>(comm.size(), 1);
    }
    std::vector<int> recv_buffer(1);
    auto [recv_count, recv_type, send_type] = comm.scatterv(
        send_buf(input),
        kamping::send_counts(send_counts),
        recv_count_out(),
        recv_buf(recv_buffer),
        recv_type_out(),
        send_type_out()
    );

    EXPECT_EQ(recv_type, MPI_INT);
    EXPECT_EQ(recv_count, 1);
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    if (comm.is_root()) {
        EXPECT_EQ(send_type, MPI_INT);
    }
}

TEST(ScattervTest, structured_bindings_implicit_recv_buf) {
    Communicator     comm;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1);
    std::vector<int> send_counts;
    if (comm.is_root()) {
        send_counts = std::vector<int>(comm.size(), 1);
    }
    auto [recv_buffer, recv_count, recv_type, send_type] = comm.scatterv(
        send_buf(input),
        kamping::send_counts(send_counts),
        recv_count_out(),
        recv_type_out(),
        send_type_out()
    );

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_type, MPI_INT);
    }
}

TEST(ScattervTest, structured_bindings_explicit_owning_recv_buf) {
    Communicator     comm;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1);
    std::vector<int> send_counts;
    if (comm.is_root()) {
        send_counts = std::vector<int>(comm.size(), 1);
    }
    auto [recv_count, recv_type, send_type, recv_buffer] = comm.scatterv(
        send_buf(input),
        kamping::send_counts(send_counts),
        recv_count_out(),
        recv_type_out(),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{})
    );

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_type, MPI_INT);
    }
}

TEST(ScattervTest, structured_bindings_explicit_owning_recv_buf_non_owning_recv_type) {
    Communicator     comm;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1);
    MPI_Datatype     recv_type;
    std::vector<int> send_counts;
    if (comm.is_root()) {
        send_counts = std::vector<int>(comm.size(), 1);
    }
    auto [recv_count, send_type, recv_buffer] = comm.scatterv(
        send_buf(input),
        recv_count_out(),
        kamping::send_counts(send_counts),
        recv_type_out(recv_type),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{})
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root()) {
        EXPECT_EQ(send_type, MPI_INT);
    }
}

TEST(ScattervTest, structured_bindings_explicit_owning_recv_buf_and_root_param) {
    Communicator     comm;
    int const        root  = comm.size_signed() - 1;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1, root);
    std::vector<int> send_counts;
    if (comm.is_root(root)) {
        send_counts = std::vector<int>(comm.size(), 1);
    }
    MPI_Datatype recv_type;
    auto [recv_count, send_type, recv_buffer, send_displs] = comm.scatterv(
        kamping::send_counts(send_counts),
        recv_type_out(recv_type),
        recv_count_out(),
        send_buf(input),
        send_type_out(),
        recv_buf<resize_to_fit>(std::vector<int>{}),
        kamping::root(root),
        send_displs_out()
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_type, MPI_INT);
    if (comm.is_root(root)) {
        EXPECT_EQ(send_type, MPI_INT);
        std::vector<int> const expected_send_displs = iota_container_n(comm.size(), 0);
        EXPECT_EQ(send_displs, expected_send_displs);
    }
}

TEST(ScattervTest, structured_bindings_explicit_non_owning_recv_buf_and_root_param_with_send_counts_out_on_non_root) {
    Communicator     comm;
    int const        root  = comm.size_signed() - 1;
    auto const       input = create_equiv_sized_input_vector_on_root(comm, 1, root);
    std::vector<int> recv_buffer;
    if (comm.is_root(root)) {
        std::vector<int> send_counts(comm.size(), 1);
        comm.scatterv(
            kamping::send_counts(send_counts),
            send_buf(input),
            kamping::root(root),
            recv_buf<resize_to_fit>(recv_buffer)
        );
    } else {
        auto [send_counts] = comm.scatterv(
            send_counts_out(),
            send_buf(input),
            kamping::root(root),
            recv_buf<resize_to_fit>(recv_buffer)
        );
        EXPECT_EQ(send_counts.size(), 0);
    }
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_signed()));
}
