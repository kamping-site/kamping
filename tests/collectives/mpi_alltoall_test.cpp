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

#include <algorithm>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/span.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AlltoallTest, single_element_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    auto mpi_result = comm.alltoall(send_buf(input), send_count_out(), recv_count_out());

    auto recv_buffer = mpi_result.extract_recv_buffer();
    auto send_count  = mpi_result.extract_send_count();
    auto recv_count  = mpi_result.extract_recv_count();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_buffer.size(), comm.size());

    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, single_element_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    std::vector<int> result;

    auto mpi_result = comm.alltoall(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        send_count_out(),
        recv_count_out()
    );
    auto send_count = mpi_result.extract_send_count();
    auto recv_count = mpi_result.extract_recv_count();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, given_recv_buffer_is_bigger_than_required) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    int const default_init_value = 42;
    auto      gen_recv_buf       = [&]() {
        return std::vector<int>(comm.size() * 2, default_init_value);
    };
    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);

    {
        // recv buffer will be resized to the number of recv elements
        auto recv_buffer = gen_recv_buf();
        EXPECT_GT(recv_buffer.size(), comm.size());
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::resize_to_fit>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_result);
    }
    {
        // recv buffer will not be resized as it is large enough
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::grow_only>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        // first half of result buffer contains recv buffer, second half remains untouched
        std::vector<int> const first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        std::vector<int> const second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
        EXPECT_EQ(first_half, expected_result);
        EXPECT_EQ(second_half, std::vector<int>(comm.size(), default_init_value));
    }
    {
        // recv buffer will not be resized
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::no_resize>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        // first half of result buffer contains recv buffer, second half remains untouched
        std::vector<int> const first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        std::vector<int> const second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
        EXPECT_EQ(first_half, expected_result);
        EXPECT_EQ(second_half, std::vector<int>(comm.size(), default_init_value));
    }
    {
        // recv buffer will not be resized as recv_buf's default resize policy is do_not_resize
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        // first half of result buffer contains recv buffer, second half remains untouched
        std::vector<int> const first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        std::vector<int> const second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
        EXPECT_EQ(first_half, expected_result);
        EXPECT_EQ(second_half, std::vector<int>(comm.size(), default_init_value));
    }
}

TEST(AlltoallTest, given_recv_buffer_is_smaller_than_required) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    int const default_init_value = 42;
    auto      gen_recv_buf       = [&]() {
        return std::vector<int>(comm.size() - 1, default_init_value);
    };
    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);

    {
        // recv buffer will be resized to the number of recv elements
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::resize_to_fit>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_result);
    }
    {
        // recv buffer will be resized as it is not large enough
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::grow_only>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_result);
    }
}

TEST(AlltoallTest, single_element_with_send_count) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    auto             mpi_result = comm.alltoall(send_buf(input), send_count(1), recv_count_out());
    std::vector<int> recv_buf   = mpi_result.extract_recv_buffer();
    int              recv_count = mpi_result.extract_recv_count();

    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_buf.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(recv_buf, expected_result);
}

TEST(AlltoallTest, single_element_with_send_and_recv_counts_out) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    // the values in send_counts_out, recv_counts_out should be ignored as they merely provide "storage" for the values
    // computed by kamping. (A mechanism which is not that useful for plain integers)
    auto             mpi_result = comm.alltoall(send_buf(input), send_count_out(), recv_count_out());
    std::vector<int> recv_buf   = mpi_result.extract_recv_buffer();
    int              send_count = mpi_result.extract_send_count();
    int              recv_count = mpi_result.extract_recv_count();

    EXPECT_EQ(recv_buf.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
}

TEST(AlltoallTest, multiple_elements) {
    Communicator comm;

    int const num_elements_per_processor_pair = 4;

    std::vector<int> input(comm.size() * num_elements_per_processor_pair);
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](int const element) -> int {
        return element / num_elements_per_processor_pair;
    });

    std::vector<int> result;
    auto             mpi_result = comm.alltoall(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        send_count_out(),
        recv_count_out()
    );

    EXPECT_EQ(mpi_result.extract_send_count(), 4);
    EXPECT_EQ(mpi_result.extract_recv_count(), 4);

    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);

    std::vector<int> expected_result(comm.size() * num_elements_per_processor_pair, comm.rank_signed());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, given_send_count_overrides_deduced_send_count) {
    Communicator comm;

    int const num_elements_per_processor_pair = 4;

    std::vector<int> input(comm.size() * num_elements_per_processor_pair);
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](int const element) -> int {
        return element / num_elements_per_processor_pair;
    });
    input.resize(input.size() * 2); // send buffer holds more elements than actually being sent
    std::vector<int> result;
    auto             mpi_result = comm.alltoall(
        send_buf(input),
        send_count(num_elements_per_processor_pair),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        recv_count_out()
    );

    EXPECT_EQ(mpi_result.extract_recv_count(), num_elements_per_processor_pair);

    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);

    std::vector<int> expected_result(comm.size() * num_elements_per_processor_pair, comm.rank_signed());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, custom_type_custom_container) {
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    OwnContainer<CustomType> input(comm.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), i};
    }

    auto recv_buf = comm.alltoall(send_buf(input), kamping::recv_buf(alloc_new<OwnContainer<CustomType>>));
    ASSERT_NE(recv_buf.data(), nullptr);
    EXPECT_EQ(recv_buf.size(), comm.size());

    OwnContainer<CustomType> expected_result(comm.size());
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {i, comm.rank()};
    }
    EXPECT_EQ(recv_buf, expected_result);
}

TEST(AlltoallTest, default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    // This just has to compile
    OwnContainer<int> result = comm.alltoall(send_buf(input));
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(AlltoallTest, given_recv_buffer_with_no_resize_policy) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> recv_buffer;
    std::vector<int> send_displs_buffer;
    std::vector<int> recv_counts_buffer;
    std::vector<int> recv_displs_buffer;
    // test kassert for sufficient size of recv buffer
    EXPECT_KASSERT_FAILS(comm.alltoall(send_buf(input), send_count(1), recv_buf<no_resize>(recv_buffer)), "");
    // same test but this time without explicit no_resize for the recv buffer as this is the default resize
    // policy
    EXPECT_KASSERT_FAILS(comm.alltoall(send_buf(input), send_count(1), recv_buf(recv_buffer)), "");
}
#endif

TEST(AlltoallTest, send_type_is_out_parameter) {
    Communicator     comm;
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> recv_buffer(comm.size(), 0);

    MPI_Datatype send_type_value;
    comm.alltoall(send_buf(input), send_type_out(send_type_value), send_count(1), recv_buf(recv_buffer));

    EXPECT_EQ(send_type_value, MPI_INT);
    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, recv_type_is_out_parameter) {
    Communicator     comm;
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> recv_buffer(comm.size(), 0);

    MPI_Datatype recv_type_value;
    comm.alltoall(send_buf(input), recv_type_out(recv_type_value), send_count(1), recv_buf(recv_buffer));

    EXPECT_EQ(recv_type_value, MPI_INT);
    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, send_recv_type_are_part_of_result_object) {
    Communicator     comm;
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> recv_buffer(comm.size(), 0);

    auto result =
        comm.alltoall(send_buf(input), send_type_out(), send_count(1), recv_type_out(), recv_buf(recv_buffer));

    EXPECT_EQ(result.extract_send_type(), MPI_INT);
    EXPECT_EQ(result.extract_recv_type(), MPI_INT);

    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, single_element_non_trivial_send_type) {
    // Each rank sends one integer (with padding) to each other rank and receives the integer without padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input(3 * comm.size());
    std::vector<int> recv_buffer(comm.size(), 0);
    for (size_t i = 0; i < comm.size(); ++i) {
        input[3 * i] = static_cast<int>(i);
    }

    MPI_Type_commit(&int_padding_padding);
    comm.alltoall(send_buf(input), send_type(int_padding_padding), send_count(1), recv_buf(recv_buffer));
    MPI_Type_free(&int_padding_padding);

    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, single_element_non_trivial_recv_type) {
    // Each rank sends one integer to each other rank and receives the integer with padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input(comm.size());
    std::vector<int> recv_buffer(3 * comm.size(), 0);
    std::iota(input.begin(), input.end(), 0);

    MPI_Type_commit(&int_padding_padding);
    comm.alltoall(send_buf(input), recv_type(int_padding_padding), recv_count(1), recv_buf(recv_buffer));
    MPI_Type_free(&int_padding_padding);

    std::vector<int> expected_result(3 * comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 3] = comm.rank_signed();
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, different_send_and_recv_counts) {
    // A rank sends two integers to each other rank. A rank receives two integers into a custom datatype with can store
    // two integers with padding
    // => recv_count == 1
    Communicator comm;
    MPI_Datatype int_padding_int = MPI_INT_padding_MPI_INT();

    std::vector<int> input(2 * comm.size());
    std::vector<int> recv_buffer(3 * comm.size(), 0);
    std::iota(input.begin(), input.end(), 0);
    int send_count_value = -1;

    MPI_Type_commit(&int_padding_int);
    comm.alltoall(
        send_buf(input),
        send_count_out(send_count_value),
        recv_type(int_padding_int),
        recv_count(1),
        recv_buf(recv_buffer)
    );
    MPI_Type_free(&int_padding_int);

    EXPECT_EQ(send_count_value, 2);

    std::vector<int> expected_result(3 * comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 3]     = comm.rank_signed() * 2;
        expected_result[i * 3 + 2] = comm.rank_signed() * 2 + 1;
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, different_send_and_recv_counts_without_explicit_mpi_types) {
    Communicator comm;
    struct CustomRecvStruct {
        int  a;
        int  b;
        bool operator==(CustomRecvStruct const& other) const {
            return std::tie(a, b) == std::tie(other.a, other.b);
        }
    };

    std::vector<int>              input(2 * comm.size());
    std::vector<CustomRecvStruct> recv_buffer(comm.size());
    std::iota(input.begin(), input.end(), 0);
    comm.alltoall(send_buf(input), recv_count(1), recv_buf(recv_buffer));
    std::vector<CustomRecvStruct> expected_result(comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i] = CustomRecvStruct{comm.rank_signed() * 2, comm.rank_signed() * 2 + 1};
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, structured_bindings_explicit_recv_buffer) {
    Communicator comm;
    // each PE sends its rank to all other PEs
    std::vector<std::uint64_t> const input(comm.size(), comm.rank());
    std::vector<std::uint64_t>       recv_buffer(comm.size());
    // explicit recv buffer
    auto [send_type, recv_type, send_count, recv_count] = comm.alltoall(
        send_type_out(),
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(recv_buffer),
        recv_type_out(),
        send_count_out(),
        recv_count_out()
    );

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_type));
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(recv_type));
    EXPECT_EQ(recv_buffer, iota_container_n<std::vector<std::uint64_t>>(comm.size(), 0ull));
}

TEST(AlltoallTest, structured_bindings_implicit_recv_buffer) {
    Communicator comm;
    // each PE sends its rank to all other PEs
    std::vector<std::uint64_t> const input(comm.size(), comm.rank());
    auto [recv_buffer, send_type, recv_type, send_count, recv_count] =
        comm.alltoall(send_type_out(), send_buf(input), recv_type_out(), send_count_out(), recv_count_out());

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_type));
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(recv_type));
    EXPECT_EQ(recv_buffer, iota_container_n<std::vector<std::uint64_t>>(comm.size(), 0ull));
}

TEST(AlltoallTest, structured_bindings_explicit_owning_recv_buffer) {
    Communicator comm;
    // each PE sends its rank to all other PEs
    std::vector<std::uint64_t> const input(comm.size(), comm.rank());
    auto [send_type, recv_buffer, recv_type, send_count, recv_count] = comm.alltoall(
        send_type_out(),
        send_buf(input),
        recv_buf(std::vector<std::uint64_t>(comm.size())),
        recv_type_out(),
        send_count_out(),
        recv_count_out()
    );

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_type));
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(recv_type));
    EXPECT_EQ(recv_buffer, iota_container_n<std::vector<std::uint64_t>>(comm.size(), 0ull));
}

TEST(AlltoallTest, inplace_basic) {
    Communicator     comm;
    std::vector<int> input(comm.size() * 2, comm.rank_signed());
    comm.alltoall(send_recv_buf(input));
    std::vector<int> expected_result(comm.size() * 2);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AlltoallTest, inplace_out_parameters) {
    Communicator     comm;
    std::vector<int> input(comm.size() * 2, comm.rank_signed());
    auto [count, type] = comm.alltoall(send_recv_buf(input), send_recv_count_out(), send_recv_type_out());
    std::vector<int> expected_result(comm.size() * 2);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    EXPECT_EQ(input, expected_result);
}

TEST(AlltoallTest, inplace_rvalue_buffer) {
    Communicator     comm;
    std::vector<int> input(comm.size() * 2, comm.rank_signed());
    auto [output, count, type] =
        comm.alltoall(send_recv_buf(std::move(input)), send_recv_count_out(), send_recv_type_out());
    std::vector<int> expected_result(comm.size() * 2);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    EXPECT_EQ(output, expected_result);
}

TEST(AlltoallTest, inplace_explicit_count) {
    Communicator comm;
    // make the buffer too big
    std::vector<int> input(comm.size() * 2 + 5, comm.rank_signed());
    comm.alltoall(send_recv_buf(input), send_recv_count(2), send_recv_type_out());
    std::vector<int> expected_result(comm.size() * 2 + 5);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    // the last 5 elements are untouched, because the buffer is not resized
    for (size_t i = comm.size() * 2; i < comm.size() * 2 + 5; ++i) {
        expected_result[i] = comm.rank_signed();
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AlltoallTest, inplace_explicit_count_resize) {
    Communicator comm;
    // make the buffer too big
    std::vector<int> input(comm.size() * 2 + 5, comm.rank_signed());
    comm.alltoall(send_recv_buf<resize_to_fit>(input), send_recv_count(2), send_recv_type_out());
    std::vector<int> expected_result(comm.size() * 2); // the buffer will be resized to only hold the received elements
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AlltoallTest, inplace_explicit_type) {
    Communicator                     comm;
    std::vector<std::pair<int, int>> input(comm.size() * 2, {comm.rank_signed(), comm.rank_signed() + 1});
    MPI_Datatype                     type = struct_type<std::pair<int, int>>::data_type();
    MPI_Type_commit(&type);
    comm.alltoall(send_recv_buf(input), send_recv_type(type), send_recv_count(2));
    MPI_Type_free(&type);
    std::vector<std::pair<int, int>> expected_result(
        comm.size() * 2
    ); // the buffer will be resized to only hold the received elements
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = {static_cast<int>(i), static_cast<int>(i + 1)};
        expected_result[i * 2 + 1] = {static_cast<int>(i), static_cast<int>(i + 1)};
    }
    EXPECT_EQ(input, expected_result);
}
