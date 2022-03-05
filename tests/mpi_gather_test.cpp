// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

// overwrite build options and set assertion level to normal, enable exceptions
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL kamping::assert::normal
#ifndef KAMPING_EXCEPTION_MODE
    #define KAMPING_EXCEPTION_MODE
#endif // KAMPING_EXCEPTION_MODE

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(GatherTest, GatherSingleElementNoReceiveBuffer) {
    Communicator comm;
    auto         value = comm.rank();

    // Test default root of communicator
    auto result = comm.gather(send_buf(value)).extract_recv_buffer();
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), 0);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.gather(send_buf(value)).extract_recv_buffer();
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (auto i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i)).extract_recv_buffer();
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), comm.size());
            for (auto j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[asserting_cast<size_t>(j)], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }

    // Check if invalid roots are identified
    for (auto i = 0; i < comm.size(); ++i) {
        EXPECT_THROW(comm.gather(send_buf(value), root(comm.size() + i)), KassertException);
    }
}

TEST(GatherTest, GatherSingleCustomElementNoReceiveBuffer) {
    Communicator comm;
    struct CustomDataType {
        int rank;
        int additional_value;
    }; // struct custom_data_type

    CustomDataType value = {comm.rank(), comm.size() - comm.rank()};

    // Test default root of communicator
    auto result = comm.gather(send_buf(value)).extract_recv_buffer();
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), 0);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)].rank, i);
            EXPECT_EQ(result[asserting_cast<size_t>(i)].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.gather(send_buf(value)).extract_recv_buffer();
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)].rank, i);
            EXPECT_EQ(result[asserting_cast<size_t>(i)].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (auto i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i)).extract_recv_buffer();
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), comm.size());
            for (auto j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[asserting_cast<size_t>(j)].rank, j);
                EXPECT_EQ(result[asserting_cast<size_t>(j)].additional_value, comm.size() - j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }

    // Check if invalid roots are identified
    for (auto i = 0; i < comm.size(); ++i) {
        EXPECT_THROW(comm.gather(send_buf(value), root(comm.size() + i)), KassertException);
    }
}

TEST(GatherTest, GatherSingleElementWithReceiveBuffer) {
    Communicator                 comm;
    auto                         value = comm.rank();
    std::vector<decltype(value)> result(0);

    // Test default root of communicator
    comm.gather(send_buf(value), recv_buf(result));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), 0);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    comm.gather(send_buf(value), recv_buf(result));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), comm.size());
        for (auto i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[asserting_cast<size_t>(i)], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (auto i = 0; i < comm.size(); ++i) {
        comm.gather(send_buf(value), recv_buf(result), root(i));
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), comm.size());
            for (auto j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[asserting_cast<size_t>(j)], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }

    // Check if invalid roots are identified
    for (auto i = 0; i < comm.size(); ++i) {
        EXPECT_THROW(comm.gather(send_buf(value), recv_buf(result), root(comm.size() + i)), KassertException);
    }

    comm.root(0);

    // receive with feasible smaller type
    std::vector<short> short_result(0);
    comm.gather(send_buf(value), recv_buf(short_result));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(comm.root(), 0);
        EXPECT_EQ(short_result.size(), 2 * comm.size());
        for (auto i = 0; i < 2 * comm.size(); ++i) {
            if (i % 2 == 0) {
                EXPECT_EQ(short_result[asserting_cast<size_t>(i)], i / 2);
            } else {
                EXPECT_EQ(short_result[asserting_cast<size_t>(i)], 0);
            }
        }
    } else {
        EXPECT_EQ(short_result.size(), 0);
    }
}

TEST(GatherTest, GatherMultipleElements) {
    Communicator comm;

    std::vector<int> values = {comm.rank(), comm.rank(), comm.rank(), comm.rank()};

    auto result = comm.gather(send_buf(values));
}
