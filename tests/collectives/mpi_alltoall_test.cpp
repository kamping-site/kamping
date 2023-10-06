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

    auto mpi_result = comm.alltoall(send_buf(input));

    auto recv_buffer = mpi_result.extract_recv_buffer();
    auto send_count  = mpi_result.extract_send_counts();
    auto recv_count  = mpi_result.extract_recv_counts();

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

    auto mpi_result = comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::resize_to_fit>(result));
    auto send_count = mpi_result.extract_send_counts();
    auto recv_count = mpi_result.extract_recv_counts();

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
        const std::vector<int> first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        const std::vector<int> second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
        EXPECT_EQ(first_half, expected_result);
        EXPECT_EQ(second_half, std::vector<int>(comm.size(), default_init_value));
    }
    {
        // recv buffer will not be resized
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::no_resize>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        // first half of result buffer contains recv buffer, second half remains untouched
        const std::vector<int> first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        const std::vector<int> second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
        EXPECT_EQ(first_half, expected_result);
        EXPECT_EQ(second_half, std::vector<int>(comm.size(), default_init_value));
    }
    {
        // recv buffer will not be resized as recv_buf's default resize policy is do_not_resize
        auto recv_buffer = gen_recv_buf();
        comm.alltoall(send_buf(input), recv_buf(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        // first half of result buffer contains recv buffer, second half remains untouched
        const std::vector<int> first_half(recv_buffer.begin(), recv_buffer.begin() + comm.size_signed());
        const std::vector<int> second_half(recv_buffer.begin() + comm.size_signed(), recv_buffer.end());
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

TEST(AlltoallTest, single_element_with_send_counts) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    auto             mpi_result = comm.alltoall(send_buf(input), send_counts(1));
    std::vector<int> recv_buf   = mpi_result.extract_recv_buffer();
    int              recv_count = mpi_result.extract_recv_counts();

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
    auto mpi_result = comm.alltoall(send_buf(input), send_counts_out(alloc_new<int>), recv_counts_out(alloc_new<int>));
    std::vector<int> recv_buf   = mpi_result.extract_recv_buffer();
    int              send_count = mpi_result.extract_send_counts();
    int              recv_count = mpi_result.extract_recv_counts();

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
    auto             mpi_result = comm.alltoall(send_buf(input), recv_buf<BufferResizePolicy::resize_to_fit>(result));

    EXPECT_EQ(mpi_result.extract_send_counts(), 4);
    EXPECT_EQ(mpi_result.extract_recv_counts(), 4);

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
        send_counts(num_elements_per_processor_pair),
        recv_buf<BufferResizePolicy::resize_to_fit>(result)
    );

    EXPECT_EQ(mpi_result.extract_recv_counts(), num_elements_per_processor_pair);

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

    auto result = comm.alltoall(send_buf(input), recv_buf(alloc_new<OwnContainer<CustomType>>)).extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size());

    OwnContainer<CustomType> expected_result(comm.size());
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {i, comm.rank()};
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    // This just has to compile
    OwnContainer<int> result = comm.alltoall(send_buf(input)).extract_recv_buffer();
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
    EXPECT_KASSERT_FAILS(comm.alltoall(send_buf(input), send_counts(1), recv_buf<no_resize>(recv_buffer)), "");
    // same test but this time without explicit no_resize for the recv buffer as this is the default resize
    // policy
    EXPECT_KASSERT_FAILS(comm.alltoall(send_buf(input), send_counts(1), recv_buf(recv_buffer)), "");
}
#endif

// ------------------------------------------------------------
// Alltoallv tests

TEST(AlltoallvTest, single_element_no_parameters) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    Communicator comm;

    // Prepare send buffer (all zeros)
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    // Prepare send counts (all ones)
    std::vector<int> send_counts(comm.size(), 1);

    // Do the alltoallv
    auto mpi_result = comm.alltoallv(send_buf(input), kamping::send_counts(send_counts));

    // Check recv buf
    auto result = mpi_result.extract_recv_buffer();
    EXPECT_EQ(result.size(), comm.size());
    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(result, expected_result);

    // Check recv counts
    auto recv_counts = mpi_result.extract_recv_counts();
    EXPECT_EQ(recv_counts, send_counts);

    // Check displacements (same for send and recv)
    std::vector<int> expected_displs(comm.size());
    std::iota(expected_displs.begin(), expected_displs.end(), 0);

    auto send_displs = mpi_result.extract_send_displs();
    EXPECT_EQ(send_displs, expected_displs);

    auto recv_displs = mpi_result.extract_recv_displs();
    EXPECT_EQ(recv_displs, expected_displs);
}

TEST(AlltoallvTest, single_element_with_receive_buffer) {
    // Sends a single element from each rank to each other rank with the recv buffer as an input parameter
    Communicator comm;

    // Prepare send buffer and counts
    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> send_counts(comm.size(), 1);

    // Do the alltoallv
    std::vector<int> result;
    comm.alltoallv(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(send_counts)
    );

    // Check recv buf
    EXPECT_EQ(result.size(), comm.size());
    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, multiple_elements_same_on_all_ranks) {
    // Sends the same amount of elements from each rank to each other rank
    Communicator comm;

    // The numer of elements to send from each rank to each rank
    int const num_elements_per_processor_pair = 4;

    // Prepare send_buffer
    std::vector<int> input(comm.size() * num_elements_per_processor_pair);
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](int const element) -> int {
        return element / num_elements_per_processor_pair;
    });

    // Calculate send counts
    std::vector<int> send_counts(comm.size(), num_elements_per_processor_pair);

    // Do the alltoallv
    std::vector<int> result;
    auto             mpi_result = comm.alltoallv(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(send_counts)
    );

    // Check recv buffer
    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);
    std::vector<int> expected_result(comm.size() * num_elements_per_processor_pair, comm.rank_signed());
    EXPECT_EQ(result, expected_result);

    // Check recv counts
    auto recv_counts = mpi_result.extract_recv_counts();
    EXPECT_EQ(recv_counts, send_counts);

    // Check displacements (same for recv and send)
    std::vector<int> expected_displs(comm.size());
    std::iota(expected_displs.begin(), expected_displs.end(), 0);
    std::transform(expected_displs.begin(), expected_displs.end(), expected_displs.begin(), [](int const value) {
        return value * num_elements_per_processor_pair;
    });

    auto send_displs = mpi_result.extract_send_displs();
    EXPECT_EQ(send_displs, expected_displs);

    auto recv_displs = mpi_result.extract_recv_displs();
    EXPECT_EQ(recv_displs, expected_displs);
}

TEST(AlltoallvTest, custom_type_custom_container) {
    // Sends a single element of a custom type in a custom container from each rank to each other rank
    Communicator comm;

    // Declare custom container
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    OwnContainer<CustomType> input(comm.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), i};
    }

    // Prepare send counts (all ones)
    std::vector<int> send_counts(comm.size(), 1);

    // Do the alltoallv - receive into a library allocated OwnContainer
    auto result = comm.alltoallv(
                          send_buf(input),
                          recv_buf(alloc_new<OwnContainer<CustomType>>),
                          kamping::send_counts(send_counts)
    )
                      .extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size());

    // Check recv buffer
    OwnContainer<CustomType> expected_result(comm.size());
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {i, comm.rank()};
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, custom_type_custom_container_i_pus_one_elements_to_rank_i) {
    // Send 1 element to rank 0, 2 elements to rank 1, ...
    // Using a custom type and container and custom containers allocated by the library for counts and displacements.
    Communicator comm;

    // Declare custom type
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    std::vector<CustomType> input((comm.size() * (comm.size() + 1) / 2));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < rank + 1; ++duplicate) {
                input[i++] = {comm.rank(), rank};
            }
        }
        ASSERT_EQ(i, input.size());
    }

    // Prepare send counts
    std::vector<int> send_counts(comm.size());
    std::iota(send_counts.begin(), send_counts.end(), 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    // Do the alltoallv - put all outputs into a custom container
    auto mpi_result = comm.alltoallv(
        send_buf(input),
        recv_buf(alloc_new<OwnContainer<CustomType>>),
        kamping::send_counts(send_counts),
        send_displs_out(alloc_new<OwnContainer<int>>),
        recv_counts_out(alloc_new<OwnContainer<int>>),
        recv_displs_out(alloc_new<OwnContainer<int>>)
    );

    // Check recv buffer
    OwnContainer<CustomType> result = mpi_result.extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size() * (comm.rank() + 1));

    OwnContainer<CustomType> expected_result(comm.size() * (comm.rank() + 1));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < comm.rank() + 1; ++duplicate) {
                expected_result[i++] = {rank, comm.rank()};
            }
        }
        ASSERT_EQ(i, expected_result.size());
    }
    EXPECT_EQ(result, expected_result);

    // Check send displs
    OwnContainer<int> send_displs = mpi_result.extract_send_displs();
    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    // Check recv counts
    OwnContainer<int> recv_counts = mpi_result.extract_recv_counts();
    OwnContainer<int> expected_recv_counts(comm.size(), comm.rank_signed() + 1);
    EXPECT_EQ(recv_counts, expected_recv_counts);

    // Check recv displs
    OwnContainer<int> recv_displs = mpi_result.extract_recv_displs();
    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one) {
    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
    // With out-parameters
    Communicator comm;

    // Declare custom type
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    std::vector<CustomType> input(comm.size() * (comm.rank() + 1));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < comm.rank() + 1; ++duplicate) {
                input[i++] = {comm.rank(), rank};
            }
        }
        ASSERT_EQ(i, input.size());
    }

    // Prepare send counts
    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    // Do the alltoallv - use output parameters
    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs;
    OwnContainer<int>        recv_counts;
    OwnContainer<int>        recv_displs;
    comm.alltoallv(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(send_counts),
        send_displs_out<BufferResizePolicy::resize_to_fit>(send_displs),
        recv_counts_out<BufferResizePolicy::resize_to_fit>(recv_counts),
        recv_displs_out<BufferResizePolicy::resize_to_fit>(recv_displs)
    );

    // Check recv buffer
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), (comm.size() * (comm.size() + 1)) / 2);

    OwnContainer<CustomType> expected_result((comm.size() * (comm.size() + 1)) / 2);
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < rank + 1; ++duplicate) {
                expected_result[i++] = {rank, comm.rank()};
            }
        }
        ASSERT_EQ(i, expected_result.size());
    }
    EXPECT_EQ(result, expected_result);

    // Check send displs
    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    // Check recv counts
    OwnContainer<int> expected_recv_counts(comm.size());
    std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 1);
    EXPECT_EQ(recv_counts, expected_recv_counts);

    // Check recv displs
    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one_given_recv_counts) {
    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
    // This time with given recv counts

    /// @todo test that no additional communication is done
    Communicator comm;

    // Declare custom type
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    std::vector<CustomType> input(comm.size() * (comm.rank() + 1));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < comm.rank() + 1; ++duplicate) {
                input[i++] = {comm.rank(), rank};
            }
        }
        ASSERT_EQ(i, input.size());
    }

    // Prepare send counts
    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    // Prepare recv counts
    OwnContainer<int> recv_counts(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 1);

    // Do the alltoallv - use output parameters
    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs;
    OwnContainer<int>        recv_displs;
    comm.alltoallv(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(send_counts),
        send_displs_out<BufferResizePolicy::resize_to_fit>(send_displs),
        kamping::recv_counts(recv_counts),
        recv_displs_out<BufferResizePolicy::resize_to_fit>(recv_displs)
    );

    // Check recv buffer
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), (comm.size() * (comm.size() + 1)) / 2);

    OwnContainer<CustomType> expected_result((comm.size() * (comm.size() + 1)) / 2);
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < rank + 1; ++duplicate) {
                expected_result[i++] = {rank, comm.rank()};
            }
        }
        ASSERT_EQ(i, expected_result.size());
    }
    EXPECT_EQ(result, expected_result);

    // Check send displs
    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    // Check recv displs
    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one_all_parameters_given) {
    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
    // This time with all parameters given

    /// @todo test that no additional communication is done
    Communicator comm;

    // Declare custom type
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    std::vector<CustomType> input(comm.size() * (comm.rank() + 1));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < comm.rank() + 1; ++duplicate) {
                input[i++] = {comm.rank(), rank};
            }
        }
        ASSERT_EQ(i, input.size());
    }

    // Prepare send counts
    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    // Prepare all counts and displacements
    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    OwnContainer<int> recv_counts(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 1);
    OwnContainer<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

    // Do the alltoallv - all counts and displacements are already pre-calculated
    comm.alltoallv(
        send_buf(input),
        recv_buf<BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(send_counts),
        kamping::send_displs(send_displs),
        kamping::recv_counts(recv_counts),
        kamping::recv_displs(recv_displs)
    );

    // Check recv buffer
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), (comm.size() * (comm.size() + 1)) / 2);

    OwnContainer<CustomType> expected_result((comm.size() * (comm.size() + 1)) / 2);
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < rank + 1; ++duplicate) {
                expected_result[i++] = {rank, comm.rank()};
            }
        }
        ASSERT_EQ(i, expected_result.size());
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, custom_type_custom_container_i_pus_one_elements_to_rank_i_all_parameters_given) {
    // Send 1 element to rank 0, 2 elements to rank 1, ...
    // This time with all parameters given

    /// @todo test that no additional communication is done
    Communicator comm;

    // Declare custom type
    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;

        bool operator==(CustomType const& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Prepare send buffer
    std::vector<CustomType> input((comm.size() * (comm.size() + 1) / 2));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < rank + 1; ++duplicate) {
                input[i++] = {comm.rank(), rank};
            }
        }
        ASSERT_EQ(i, input.size());
    }

    // Prepare all counts and displacements
    std::vector<int> send_counts(comm.size());
    std::iota(send_counts.begin(), send_counts.end(), 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    OwnContainer<int> send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);

    OwnContainer<int> recv_counts(comm.size(), comm.rank_signed() + 1);

    OwnContainer<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

    // Do the alltoallv - all counts and displacements are already pre-calculated
    auto mpi_result = comm.alltoallv(
        send_buf(input),
        recv_buf(alloc_new<OwnContainer<CustomType>>),
        kamping::send_counts(send_counts),
        kamping::send_displs(send_displs),
        kamping::recv_counts(recv_counts),
        kamping::recv_displs(recv_displs)
    );

    // Check recv buffer
    OwnContainer<CustomType> result = mpi_result.extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size() * (comm.rank() + 1));

    OwnContainer<CustomType> expected_result(comm.size() * (comm.rank() + 1));
    {
        size_t i = 0;
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            for (size_t duplicate = 0; duplicate < comm.rank() + 1; ++duplicate) {
                expected_result[i++] = {rank, comm.rank()};
            }
        }
        ASSERT_EQ(i, expected_result.size());
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, default_container_type) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    Communicator<OwnContainer> comm;

    // Prepare send buffer (all zeros)
    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    // Prepare send counts (all ones)
    std::vector<int> send_counts(comm.size(), 1);

    // Do the alltoallv
    auto mpi_result = comm.alltoallv(send_buf(input), kamping::send_counts(send_counts));

    // These just have to compile
    OwnContainer<int> result      = mpi_result.extract_recv_buffer();
    OwnContainer<int> recv_counts = mpi_result.extract_recv_counts();
    OwnContainer<int> send_displs = mpi_result.extract_send_displs();
    OwnContainer<int> recv_displs = mpi_result.extract_recv_displs();
}

TEST(AlltoallvTest, given_buffers_are_bigger_than_required) {
    // Check that if preallocated buffer are given for *_counts and *displacements that the resizing happens according
    // to the resizing policy.
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> send_counts_buffer(comm.size(), 1);

    int const        default_init_value = 42;
    std::vector<int> expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);
    std::vector<int> expected_recv_counts(comm.size(), 1);
    std::vector<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts_buffer.begin(), send_counts_buffer.end(), expected_send_displs.begin(), 0);
    std::vector<int> expected_recv_displs = expected_send_displs;

    {
        // buffers will be resized to the size of the communicator
        std::vector<int> recv_buffer(2 * comm.size(), default_init_value);
        std::vector<int> send_displs_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_counts_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_displs_buffer(2 * comm.size(), default_init_value);
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<BufferResizePolicy::resize_to_fit>(send_displs_buffer),
            recv_counts_out<BufferResizePolicy::resize_to_fit>(recv_counts_buffer),
            recv_displs_out<BufferResizePolicy::resize_to_fit>(recv_displs_buffer),
            recv_buf<BufferResizePolicy::resize_to_fit>(recv_buffer)
        );
        EXPECT_EQ(send_displs_buffer, expected_send_displs);
        EXPECT_EQ(recv_counts_buffer, expected_recv_counts);
        EXPECT_EQ(recv_displs_buffer, expected_recv_displs);
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // buffers will not be resized as they are large enough
        std::vector<int> recv_buffer(2 * comm.size(), default_init_value);
        std::vector<int> send_displs_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_counts_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_displs_buffer(2 * comm.size(), default_init_value);
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<BufferResizePolicy::grow_only>(send_displs_buffer),
            recv_counts_out<BufferResizePolicy::grow_only>(recv_counts_buffer),
            recv_displs_out<BufferResizePolicy::grow_only>(recv_displs_buffer),
            recv_buf<BufferResizePolicy::grow_only>(recv_buffer)
        );
        EXPECT_EQ(send_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(send_displs_buffer.data(), comm.size()), ElementsAreArray(expected_send_displs));
        EXPECT_THAT(Span(recv_counts_buffer.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs_buffer.data(), comm.size()), ElementsAreArray(expected_recv_displs));
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
    {
        // buffers will not be resized as the (implicit) resize policy is no_resize
        std::vector<int> recv_buffer(2 * comm.size(), default_init_value);
        std::vector<int> send_displs_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_counts_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_displs_buffer(2 * comm.size(), default_init_value);
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<BufferResizePolicy::no_resize>(send_displs_buffer),
            recv_counts_out<BufferResizePolicy::no_resize>(recv_counts_buffer),
            recv_displs_out<BufferResizePolicy::no_resize>(recv_displs_buffer),
            recv_buf<BufferResizePolicy::no_resize>(recv_buffer)
        );
        EXPECT_EQ(send_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(send_displs_buffer.data(), comm.size()), ElementsAreArray(expected_send_displs));
        EXPECT_THAT(Span(recv_counts_buffer.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs_buffer.data(), comm.size()), ElementsAreArray(expected_recv_displs));
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
    {
        // buffers will not be resized as the (implicit) resize policy is no_resize
        std::vector<int> recv_buffer(2 * comm.size(), default_init_value);
        std::vector<int> send_displs_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_counts_buffer(2 * comm.size(), default_init_value);
        std::vector<int> recv_displs_buffer(2 * comm.size(), default_init_value);
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out(send_displs_buffer),
            recv_counts_out(recv_counts_buffer),
            recv_displs_out(recv_displs_buffer),
            recv_buf(recv_buffer)
        );
        EXPECT_EQ(send_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(send_displs_buffer.data(), comm.size()), ElementsAreArray(expected_send_displs));
        EXPECT_THAT(Span(recv_counts_buffer.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs_buffer.data(), comm.size()), ElementsAreArray(expected_recv_displs));
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
}

TEST(AlltoallvTest, given_buffers_are_smaller_than_required) {
    // If preallocated buffer are given for *_counts and *displacements then check that the resizing happens according
    // to the resizing policy.
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> send_counts_buffer(comm.size(), 1);

    std::vector<int> expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);
    std::vector<int> expected_recv_counts(comm.size(), 1);
    std::vector<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts_buffer.begin(), send_counts_buffer.end(), expected_send_displs.begin(), 0);
    std::vector<int> expected_recv_displs = expected_send_displs;

    {
        // buffers will be resized to the size of the communicator
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<BufferResizePolicy::resize_to_fit>(send_displs_buffer),
            recv_counts_out<BufferResizePolicy::resize_to_fit>(recv_counts_buffer),
            recv_displs_out<BufferResizePolicy::resize_to_fit>(recv_displs_buffer),
            recv_buf<BufferResizePolicy::resize_to_fit>(recv_buffer)
        );
        EXPECT_EQ(send_displs_buffer, expected_send_displs);
        EXPECT_EQ(recv_counts_buffer, expected_recv_counts);
        EXPECT_EQ(recv_displs_buffer, expected_recv_displs);
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // buffers will not be resized as they are large enough
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<BufferResizePolicy::grow_only>(send_displs_buffer),
            recv_counts_out<BufferResizePolicy::grow_only>(recv_counts_buffer),
            recv_displs_out<BufferResizePolicy::grow_only>(recv_displs_buffer),
            recv_buf<BufferResizePolicy::grow_only>(recv_buffer)
        );
        EXPECT_THAT(Span(send_displs_buffer.data(), comm.size()), ElementsAreArray(expected_send_displs));
        EXPECT_THAT(Span(recv_counts_buffer.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs_buffer.data(), comm.size()), ElementsAreArray(expected_recv_displs));
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
}

TEST(AlltoallvTest, non_monotonically_increasing_recv_displacements) {
    // Rank i sends its rank j times to rank j. Rank i receives j's message at position comm.size() - (j + 1)*i via
    // explicit recv_displs. E.g. on rank 2 we expect recv buffer = [(size-1),(size-1), (size-2),(size-2), ..., 0, 0]
    Communicator comm;

    // prepare send buffer
    int              num_elems_to_send = (comm.size_signed() * (comm.size_signed() - 1)) / 2; // gauss' sum formula
    std::vector<int> input(static_cast<size_t>(num_elems_to_send), comm.rank_signed());

    // prepare send counts
    std::vector<int> send_counts(comm.size());
    std::iota(send_counts.begin(), send_counts.end(), 0);

    // prepare recv counts and displs
    std::vector<int> recv_counts(comm.size(), comm.rank_signed());
    std::vector<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0u);
    std::reverse(recv_displs.begin(), recv_displs.end());

    auto expected_recv_buffer = [&]() {
        std::vector<int> expected_recv_buf;
        for (int i = 0; i < comm.size_signed(); ++i) {
            int source_rank = comm.size_signed() - 1 - i;
            std::fill_n(std::back_inserter(expected_recv_buf), comm.rank(), source_rank);
        }
        return expected_recv_buf;
    };

    {
        // do the alltoallv without recv_counts
        auto recv_buf =
            comm.alltoallv(send_buf(input), kamping::send_counts(send_counts), kamping::recv_displs(recv_displs))
                .extract_recv_buffer();

        EXPECT_EQ(recv_buf, expected_recv_buffer());
    }
    {
        // do the alltoallv with recv_counts
        auto recv_buf = comm.alltoallv(
                                send_buf(input),
                                kamping::send_counts(send_counts),
                                kamping::recv_counts(recv_counts),
                                kamping::recv_displs(recv_displs)
        )
                            .extract_recv_buffer();
        EXPECT_EQ(recv_buf, expected_recv_buffer());
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(AlltoallvTest, given_buffers_are_smaller_than_required_with_no_resize_policy) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> send_counts_buffer(comm.size(), 1);

    {
        // no kasserts fail
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        comm.alltoallv(
            send_buf(input),
            send_counts(send_counts_buffer),
            send_displs_out<resize_to_fit>(send_displs_buffer),
            recv_counts_out<resize_to_fit>(recv_counts_buffer),
            recv_displs_out<resize_to_fit>(recv_displs_buffer),
            recv_buf<resize_to_fit>(recv_buffer)
        );
    }
    {
        // test kassert for recv_buffer
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf<no_resize>(recv_buffer)
            ),
            ""
        );
        // same check but this time without explicit no_resize for the recv buffer as this is the default resize policy
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf(recv_buffer)
            ),
            ""
        );
    }
    {
        // test kassert for recv_displs
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out<no_resize>(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
        // same check but this time without explicit no_resize for the recv displs buffer as this is the default resize
        // policy
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
    }
    {
        // test kassert for recv_counts
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out<no_resize>(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
        // same check but this time without explicit no_resize for the recv counts buffer as this is the default resize
        // policy
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<resize_to_fit>(send_displs_buffer),
                recv_counts_out(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
    }
    {
        // test kassert for send_displs
        std::vector<int> recv_buffer;
        std::vector<int> send_displs_buffer;
        std::vector<int> recv_counts_buffer;
        std::vector<int> recv_displs_buffer;
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out<no_resize>(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
        // same check but this time without explicit no_resize for the send displs buffer as this is the default resize
        // policy
        EXPECT_KASSERT_FAILS(
            comm.alltoallv(
                send_buf(input),
                send_counts(send_counts_buffer),
                send_displs_out(send_displs_buffer),
                recv_counts_out<resize_to_fit>(recv_counts_buffer),
                recv_displs_out<resize_to_fit>(recv_displs_buffer),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
    }
}
#endif
