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

#include <algorithm>
#include <numeric>

#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"

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

    auto mpi_result = comm.alltoall(send_buf(input), recv_buf(result));
    auto send_count = mpi_result.extract_send_counts();
    auto recv_count = mpi_result.extract_recv_counts();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
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

TEST(AlltoallTest, single_element_with_send_and_recv_counts) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    auto             mpi_result = comm.alltoall(send_buf(input), send_counts(1), recv_counts(1));
    std::vector<int> recv_buf   = mpi_result.extract_recv_buffer();

    EXPECT_EQ(recv_buf.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(recv_buf, expected_result);
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
    auto             mpi_result = comm.alltoall(send_buf(input), recv_buf(result));

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
    auto mpi_result = comm.alltoall(send_buf(input), send_counts(num_elements_per_processor_pair), recv_buf(result));

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
    comm.alltoall(send_buf(input), send_type(int_padding_padding), send_counts(1), recv_buf(recv_buffer));
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
    comm.alltoall(send_buf(input), recv_type(int_padding_padding), recv_buf(recv_buffer));
    MPI_Type_free(&int_padding_padding);

    std::vector<int> expected_result(3 * comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 3] = comm.rank_signed();
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallTest, different_send_and_recv_counts) {
    // Rank i sends
    Communicator comm;
    MPI_Datatype int_padding_int = MPI_INT_padding_MPI_INT();

    std::vector<int> input(2 * comm.size());
    std::vector<int> recv_buffer(3 * comm.size(), 0);
    std::iota(input.begin(), input.end(), 0);

    MPI_Type_commit(&int_padding_int);
    auto mpi_result = comm.alltoall(send_buf(input), recv_counts(1), recv_type(int_padding_int), recv_buf(recv_buffer));
    MPI_Type_free(&int_padding_int);

    auto send_count = mpi_result.extract_send_counts();

    EXPECT_EQ(send_count, 2);

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
    comm.alltoall(send_buf(input), recv_counts(1), recv_buf(recv_buffer));
    std::vector<CustomRecvStruct> expected_result(comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i] = CustomRecvStruct{comm.rank_signed() * 2, comm.rank_signed() * 2 + 1};
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

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
    comm.alltoallv(send_buf(input), recv_buf(result), kamping::send_counts(send_counts));

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
    auto             mpi_result = comm.alltoallv(send_buf(input), recv_buf(result), kamping::send_counts(send_counts));

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
        recv_buf(result),
        kamping::send_counts(send_counts),
        send_displs_out(send_displs),
        recv_counts_out(recv_counts),
        recv_displs_out(recv_displs)
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
        recv_buf(result),
        kamping::send_counts(send_counts),
        send_displs_out(send_displs),
        kamping::recv_counts(recv_counts),
        recv_displs_out(recv_displs)
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
        recv_buf(result),
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

TEST(AlltoallvTest, receive_msg_in_reverse_order) {
    // Rank i sends its rank j times to rank j. Rank i receives j's message at position comm.size() - (j + 1) via
    // explicit recv_displs.
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

TEST(AlltoallvTest, single_element_non_trivial_send_type) {
    // Each rank sends one integer (with padding) to each other rank and receives the integer without padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input(3 * comm.size());
    std::vector<int> recv_buffer(comm.size(), 0);
    for (size_t i = 0; i < comm.size(); ++i) {
        input[3 * i] = static_cast<int>(i);
    }
    std::vector<int> send_counts(comm.size(), 1);

    MPI_Type_commit(&int_padding_padding);
    comm.alltoallv(
        send_buf(input),
        send_type(int_padding_padding),
        kamping::send_counts(send_counts),
        recv_buf(recv_buffer)
    );
    MPI_Type_free(&int_padding_padding);

    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AlltoallvTest, multiple_elements_non_trivial_recv_type) {
    // Rank i sends i integers (without padding) to each other rank. Each rank receives the sent integers with padding.
    Communicator comm;
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();

    std::vector<int> input(comm.size() * comm.rank(), comm.rank_signed());
    std::vector<int> send_counts(comm.size(), comm.rank_signed());
    size_t const     num_elems_to_recv =
        static_cast<size_t>((comm.size_signed() * (comm.size_signed() - 1)) / 2); // gauss' sum formula
    std::vector<int> recv_buffer(3 * num_elems_to_recv, 0);

    MPI_Type_commit(&int_padding_padding);
    comm.alltoallv(
        send_buf(input),
        kamping::send_counts(send_counts),
        recv_type(int_padding_padding),
        recv_buf(recv_buffer)
    );
    MPI_Type_free(&int_padding_padding);

    std::vector<int> expected_result(3 * num_elems_to_recv);
    size_t           base_idx = 0;
    for (size_t i = 1; i < comm.size(); ++i) {
        base_idx += (i - 1) * 3;
        for (size_t j = 0; j < i; ++j) {
            expected_result[base_idx + j * 3] = static_cast<int>(i);
        }
    }
    EXPECT_EQ(recv_buffer, expected_result);
}
