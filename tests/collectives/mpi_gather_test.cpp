// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <cstddef>

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(GatherTest, gather_single_element_no_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    // Test default root of communicator
    auto result = comm.gather(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    result = comm.gather(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i)).extract_recv_buffer();
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_single_custom_element_no_receive_buffer) {
    Communicator comm;
    struct CustomDataType {
        int rank;
        int additional_value;
    }; // struct custom_data_type

    CustomDataType value = {comm.rank_signed(), comm.size_signed() - comm.rank_signed()};

    // Test default root of communicator
    auto result = comm.gather(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i].rank, i);
            EXPECT_EQ(result[i].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    result = comm.gather(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i].rank, i);
            EXPECT_EQ(result[i].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i)).extract_recv_buffer();
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j].rank, j);
                EXPECT_EQ(result[j].additional_value, comm.size() - j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_single_element_with_receive_buffer) {
    Communicator                 comm;
    auto                         value = comm.rank();
    std::vector<decltype(value)> result(0);

    // Test default root of communicator
    comm.gather(send_buf(value), recv_buf(result));
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    comm.gather(send_buf(value), recv_buf(result));
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        comm.gather(send_buf(value), recv_buf(result), root(i));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_multiple_elements_no_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    auto             result = comm.gather(send_buf(values)).extract_recv_buffer();

    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.gather(send_buf(values)).extract_recv_buffer();
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(values), root(i)).extract_recv_buffer();
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            EXPECT_EQ(result.size(), values.size() * comm.size());
            for (size_t j = 0; j < result.size(); ++j) {
                EXPECT_EQ(result[j], j / values.size());
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_multiple_elements_with_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    std::vector<int> result(0);

    comm.gather(send_buf(values), recv_buf(result));

    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    comm.gather(send_buf(values), recv_buf(result));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        comm.gather(send_buf(values), root(i), recv_buf(result));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            EXPECT_EQ(result.size(), values.size() * comm.size());
            for (size_t j = 0; j < result.size(); ++j) {
                EXPECT_EQ(result[j], j / values.size());
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_receive_custom_container) {
    Communicator      comm;
    std::vector<int>  values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    OwnContainer<int> result;

    comm.gather(send_buf(values), recv_buf(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_send_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    std::vector<int> result;

    comm.gather(send_buf(values), recv_buf(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_send_and_receive_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    OwnContainer<int> result;

    comm.gather(send_buf(values), recv_buf(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}
