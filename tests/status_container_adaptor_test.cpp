// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>

#include "kamping/status_container_adaptor.hpp"

TEST(StatusContainerAdaptorTest, empty) {
    std::vector<MPI_Status>           statuses;
    kamping::status_container_adaptor s(statuses);
    EXPECT_EQ(s.size(), 0);
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.begin(), s.end());
}

TEST(StatusContainerAdaptorTest, basics) {
    std::vector<MPI_Status> statuses(4);
    for (size_t i = 0; i < statuses.size(); ++i) {
        statuses[i].MPI_SOURCE = static_cast<int>(i);
        statuses[i].MPI_TAG    = static_cast<int>(i);
        MPI_Status_set_elements(&statuses[i], MPI_INT, static_cast<int>(i));
    }
    kamping::status_container_adaptor s(statuses);
    EXPECT_EQ(s.size(), 4);
    size_t i = 0;

    // iterator access
    auto it = s.begin();
    for (; it != s.end(); ++it) {
        kamping::StatusConstRef status = *it;
        EXPECT_EQ(status.source(), i);
        EXPECT_EQ(status.tag(), i);
        EXPECT_EQ(status.count<int>(), i);
        i++;
    }
    EXPECT_EQ(i, 4);
    EXPECT_EQ(it, s.end());
    EXPECT_EQ(std::distance(s.begin(), s.end()), 4);

    // random access
    kamping::StatusConstRef status = s[2];

    EXPECT_EQ(status.source(), 2);
    EXPECT_EQ(status.tag(), 2);
    EXPECT_EQ(status.count<int>(), 2);
}
