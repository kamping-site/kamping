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
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AlltoallTest, single_element_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    auto result = comm.alltoall(send_buf(input)).extract_recv_buffer();

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, single_element_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());

    std::vector<int> result;
    comm.alltoall(send_buf(input), recv_buf(result));

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, multiple_elements) {
    Communicator comm;

    const int num_elements_per_processor_pair = 4;

    std::vector<int> input(comm.size() * num_elements_per_processor_pair);
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](const int element) -> int {
        return element / num_elements_per_processor_pair;
    });

    std::vector<int> result;
    comm.alltoall(send_buf(input), recv_buf(result));

    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);

    std::vector<int> expected_result(comm.size() * num_elements_per_processor_pair, comm.rank_signed());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, custom_type_custom_container) {
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    OwnContainer<CustomType> input(comm.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), i};
    }

    auto result =
        comm.alltoall(send_buf(input), recv_buf(NewContainer<OwnContainer<CustomType>>{})).extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size());

    OwnContainer<CustomType> expected_result(comm.size());
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {i, comm.rank()};
    }
    EXPECT_EQ(result, expected_result);
}

// ------------------------------------------------------------
// Alltoallv tests

TEST(AlltoallvTest, single_element_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto mpi_result = comm.alltoallv(send_buf(input), kamping::send_counts(send_counts));

    auto result = mpi_result.extract_recv_buffer();
    EXPECT_EQ(result.size(), comm.size());
    std::vector<int> expected_result(comm.size(), comm.rank_signed());
    EXPECT_EQ(result, expected_result);

    auto recv_counts = mpi_result.extract_recv_counts();
    EXPECT_EQ(recv_counts, send_counts);

    std::vector<int> expected_displs(comm.size());
    std::iota(expected_displs.begin(), expected_displs.end(), 0);

    auto send_displs = mpi_result.extract_send_displs();
    EXPECT_EQ(send_displs, expected_displs);

    auto recv_displs = mpi_result.extract_recv_displs();
    EXPECT_EQ(recv_displs, expected_displs);
}

TEST(AlltoallvTest, single_element_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed());
    std::vector<int> send_counts(comm.size(), 1);

    std::vector<int> result;
    comm.alltoallv(send_buf(input), recv_buf(result), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, multiple_elements_same_on_all_ranks) {
    Communicator comm;

    int const num_elements_per_processor_pair = 4;

    std::vector<int> input(comm.size() * num_elements_per_processor_pair);
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](int const element) -> int {
        return element / num_elements_per_processor_pair;
    });

    std::vector<int> send_counts(comm.size(), num_elements_per_processor_pair);

    std::vector<int> result;
    auto             mpi_result = comm.alltoallv(send_buf(input), recv_buf(result), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);
    std::vector<int> expected_result(comm.size() * num_elements_per_processor_pair, comm.rank_signed());
    EXPECT_EQ(result, expected_result);

    auto recv_counts = mpi_result.extract_recv_counts();
    EXPECT_EQ(recv_counts, send_counts);

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
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    OwnContainer<CustomType> input(comm.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), i};
    }

    std::vector<int> send_counts(comm.size(), 1);

    auto result =
        comm.alltoallv(
                send_buf(input), recv_buf(NewContainer<OwnContainer<CustomType>>{}), kamping::send_counts(send_counts))
            .extract_recv_buffer();
    ASSERT_NE(result.data(), nullptr);
    EXPECT_EQ(result.size(), comm.size());

    OwnContainer<CustomType> expected_result(comm.size());
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {i, comm.rank()};
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallvTest, custom_type_custom_container_i_pus_one_elements_to_rank_i) {
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Send 1 element to rank 0, 2 elements to rank 1, ...
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

    std::vector<int> send_counts(comm.size());
    std::iota(send_counts.begin(), send_counts.end(), 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    auto mpi_result = comm.alltoallv(
        send_buf(input), recv_buf(NewContainer<OwnContainer<CustomType>>{}), kamping::send_counts(send_counts),
        send_displs_out(NewContainer<OwnContainer<int>>{}), recv_counts_out(NewContainer<OwnContainer<int>>{}),
        recv_displs_out(NewContainer<OwnContainer<int>>{}));

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

    OwnContainer<int> send_displs = mpi_result.extract_send_displs();
    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    OwnContainer<int> recv_counts = mpi_result.extract_recv_counts();
    OwnContainer<int> expected_recv_counts(comm.size(), comm.rank_signed() + 1);
    EXPECT_EQ(recv_counts, expected_recv_counts);

    OwnContainer<int> recv_displs = mpi_result.extract_recv_displs();
    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one) {
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
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

    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs;
    OwnContainer<int>        recv_counts;
    OwnContainer<int>        recv_displs;
    comm.alltoallv(
        send_buf(input), recv_buf(result), kamping::send_counts(send_counts), send_displs_out(send_displs),
        recv_counts_out(recv_counts), recv_displs_out(recv_displs));

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

    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    OwnContainer<int> expected_recv_counts(comm.size());
    std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 1);
    EXPECT_EQ(recv_counts, expected_recv_counts);

    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one_given_recv_counts) {
    /// @todo test that no additional communication is done
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
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

    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs;
    OwnContainer<int>        recv_counts(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 1);
    OwnContainer<int> recv_displs;
    comm.alltoallv(
        send_buf(input), recv_buf(result), kamping::send_counts(send_counts), send_displs_out(send_displs),
        kamping::recv_counts(recv_counts), recv_displs_out(recv_displs));

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

    OwnContainer<int> expected_send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), expected_send_displs.begin(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);

    OwnContainer<int> expected_recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), expected_recv_displs.begin(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvTest, custom_type_custom_container_rank_i_sends_i_plus_one_all_parameters_given) {
    /// @todo test that no additional communication is done
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Rank 0 send 1 element to each other rank, rank 1 sends 2 elements ...
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

    OwnContainer<int> send_counts(comm.size(), comm.rank_signed() + 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    OwnContainer<CustomType> result;
    OwnContainer<int>        send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    OwnContainer<int> recv_counts(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 1);
    OwnContainer<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
    comm.alltoallv(
        send_buf(input), recv_buf(result), kamping::send_counts(send_counts), kamping::send_displs(send_displs),
        kamping::recv_counts(recv_counts), kamping::recv_displs(recv_displs));

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
    /// @todo test that no additional communication is done
    Communicator comm;

    struct CustomType {
        size_t sendingRank;
        size_t receivingRank;
        bool   operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    // Send 1 element to rank 0, 2 elements to rank 1, ...
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

    std::vector<int> send_counts(comm.size());
    std::iota(send_counts.begin(), send_counts.end(), 1);
    ASSERT_EQ(std::accumulate(send_counts.begin(), send_counts.end(), 0), input.size());

    OwnContainer<int> send_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);

    OwnContainer<int> recv_counts(comm.size(), comm.rank_signed() + 1);

    OwnContainer<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

    auto mpi_result = comm.alltoallv(
        send_buf(input), recv_buf(NewContainer<OwnContainer<CustomType>>{}), kamping::send_counts(send_counts),
        kamping::send_displs(send_displs), kamping::recv_counts(recv_counts), kamping::recv_displs(recv_displs));

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
