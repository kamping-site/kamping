// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>
#include <kamping/mpi_datatype.hpp>

size_t       commit_count           = 0;
MPI_Datatype last_commited_datatype = MPI_DATATYPE_NULL;

int MPI_Type_commit(MPI_Datatype* datatype) {
    commit_count++;
    last_commited_datatype = *datatype;
    return MPI_SUCCESS;
}

size_t       free_count          = 0;
MPI_Datatype last_freed_datatype = MPI_DATATYPE_NULL;

int MPI_Type_free(MPI_Datatype* datatype) {
    free_count++;
    last_freed_datatype = *datatype;
    return MPI_SUCCESS;
}

class ScopedTypeTest : public ::testing::Test {
public:
    void SetUp() override {
        commit_count           = 0;
        last_commited_datatype = MPI_DATATYPE_NULL;
        free_count             = 0;
        last_freed_datatype    = MPI_DATATYPE_NULL;
    }
};

TEST_F(ScopedTypeTest, test_scoped_type) {
    {
        kamping::ScopedDatatype scoped_type(MPI_INT);
        auto                    type = scoped_type.data_type();
        EXPECT_EQ(type, MPI_INT);
        EXPECT_EQ(commit_count, 1);
        EXPECT_EQ(last_commited_datatype, MPI_INT);
        EXPECT_EQ(free_count, 0);

        // Test move constructor
        kamping::ScopedDatatype scoped_type2(std::move(scoped_type));
        type = scoped_type2.data_type();
        EXPECT_EQ(type, MPI_INT);
        EXPECT_EQ(commit_count, 1);
        EXPECT_EQ(last_commited_datatype, MPI_INT);
        EXPECT_EQ(free_count, 0);
    }
    EXPECT_EQ(free_count, 1);
    EXPECT_EQ(last_freed_datatype, MPI_INT);
}
TEST_F(ScopedTypeTest, test_scoped_null) {
    {
        kamping::ScopedDatatype scoped_type;
        auto                    type = scoped_type.data_type();
        EXPECT_EQ(type, MPI_DATATYPE_NULL);
        EXPECT_EQ(commit_count, 0);
        EXPECT_EQ(free_count, 0);

        // Test move constructor
        kamping::ScopedDatatype scoped_type2(std::move(scoped_type));
        type = scoped_type2.data_type();
        EXPECT_EQ(type, MPI_DATATYPE_NULL);
        EXPECT_EQ(commit_count, 0);
        EXPECT_EQ(free_count, 0);
    }
    EXPECT_EQ(free_count, 0);
}
