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

#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/communicator.hpp"
#include "kamping/group.hpp"

TEST(GroupTest, basics) {
    using namespace kamping;

    Communicator<std::vector> comm;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // Construction
    auto empty_group = Group::empty();
    auto world_group = comm_world().group();
    EXPECT_EQ(Group(MPI_GROUP_EMPTY).compare(empty_group), GroupEquality::Identical);
    EXPECT_EQ(Group(kamping::comm_world()).compare(world_group), GroupEquality::Identical);
    EXPECT_EQ(comm.group().compare(world_group), GroupEquality::Identical);
    EXPECT_EQ(Group(comm).compare(comm.group()), GroupEquality::Identical);

    EXPECT_TRUE(Group(MPI_GROUP_EMPTY).is_identical(empty_group));
    EXPECT_TRUE(Group(kamping::comm_world()).is_identical(world_group));
    EXPECT_TRUE(comm.group().is_identical(world_group));
    EXPECT_TRUE(Group(comm).is_identical(comm.group()));

    EXPECT_TRUE(Group(MPI_GROUP_EMPTY).has_same_ranks(empty_group));
    EXPECT_TRUE(Group(kamping::comm_world()).has_same_ranks(world_group));
    EXPECT_TRUE(comm.group().has_same_ranks(world_group));
    EXPECT_TRUE(Group(comm).has_same_ranks(comm.group()));

    // rank() and size()
    EXPECT_EQ(empty_group.size(), 0);
    EXPECT_EQ(world_group.size(), comm.size());
    EXPECT_EQ(world_group.rank(), comm.rank());

    // compare()
    EXPECT_TRUE(world_group.has_same_ranks(world_group));
    EXPECT_TRUE(empty_group.has_same_ranks(empty_group));

    // difference()
    auto world_empty_diff = world_group.difference(empty_group);
    auto empty_world_diff = empty_group.difference(world_group);
    auto world_world_diff = world_group.difference(world_group);
    auto empty_empty_diff = empty_group.difference(empty_group);
    EXPECT_TRUE(world_empty_diff.has_same_ranks(world_group));
    EXPECT_TRUE(empty_world_diff.has_same_ranks(empty_group));
    EXPECT_TRUE(world_world_diff.has_same_ranks(empty_group));
    EXPECT_TRUE(empty_empty_diff.has_same_ranks(empty_group));

    // intersection()
    auto world_empty_inter = world_group.intersection(empty_group);
    auto empty_world_inter = empty_group.intersection(world_group);
    auto world_world_inter = world_group.intersection(world_group);
    auto empty_empty_inter = empty_group.intersection(empty_group);
    EXPECT_TRUE(world_empty_inter.has_same_ranks(empty_group));
    EXPECT_TRUE(empty_world_inter.has_same_ranks(empty_group));
    EXPECT_TRUE(empty_empty_inter.has_same_ranks(empty_group));
    EXPECT_TRUE(world_world_inter.has_same_ranks(world_group));

    // set_union()
    auto world_empty_union = world_group.set_union(empty_group);
    auto empty_world_union = empty_group.set_union(world_group);
    auto world_world_union = world_group.set_union(world_group);
    auto empty_empty_union = empty_group.set_union(empty_group);
    EXPECT_TRUE(world_empty_union.has_same_ranks(world_group));
    EXPECT_TRUE(empty_world_union.has_same_ranks(world_group));
    EXPECT_TRUE(empty_empty_union.has_same_ranks(empty_group));
    EXPECT_TRUE(world_world_union.has_same_ranks(world_group));

    // Move assignment and move-copying
    Group world_group_copy = Group::empty();
    world_group_copy       = std::move(world_group);
    auto world_group_copy2(std::move(world_group_copy));
    EXPECT_EQ(world_group_copy2.size(), comm.size());
    EXPECT_EQ(world_group_copy2.rank(), comm.rank());
    EXPECT_TRUE(world_group_copy2.has_same_ranks(comm.group()));
}
