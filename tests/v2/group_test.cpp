#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/comm.hpp"
#include "mpi/group.hpp"
#include "mpi/handle.hpp"

using namespace mpi::experimental;

// ── Concept checks ────────────────────────────────────────────────────────────

static_assert(convertible_to_mpi_handle<group_view, MPI_Group>);
static_assert(convertible_to_mpi_handle<group, MPI_Group>);
static_assert(convertible_to_mpi_handle<comm_view, MPI_Comm>);
static_assert(convertible_to_mpi_handle<comm, MPI_Comm>);

// ── group_view ────────────────────────────────────────────────────────────────

TEST(GroupViewTest, WrapsHandle) {
    // MPI_COMM_WORLD group as a raw handle for view testing
    MPI_Group world_group = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    group_view gv(world_group);
    EXPECT_EQ(gv.mpi_handle(), world_group);
    EXPECT_EQ(gv.native(), world_group);

    MPI_Group_free(&world_group);
}

// ── group ─────────────────────────────────────────────────────────────────────

TEST(GroupTest, EmptyGroup) {
    group g = group::empty();
    EXPECT_EQ(g.mpi_handle(), MPI_GROUP_EMPTY);
}

TEST(GroupTest, FromNative) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);
    EXPECT_NE(raw, MPI_GROUP_EMPTY);

    group g = group::from_native(raw);
    EXPECT_EQ(g.mpi_handle(), raw);
    // g owns raw; it will be freed by g's destructor.
}

TEST(GroupTest, MoveConstructTransfersOwnership) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);

    group a = group::from_native(raw);
    group b = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_GROUP_EMPTY); // moved-from is MPI_GROUP_EMPTY
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(GroupTest, MoveAssignTransfersOwnership) {
    MPI_Group raw_a = MPI_GROUP_EMPTY;
    MPI_Group raw_b = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw_a);
    MPI_Comm_group(MPI_COMM_WORLD, &raw_b);

    group a = group::from_native(raw_a);
    group b = group::from_native(raw_b);
    b       = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_GROUP_EMPTY);
    EXPECT_EQ(b.mpi_handle(), raw_a);
    // raw_b was freed by b's move assignment.
}

TEST(GroupTest, ImplicitConversionToView) {
    MPI_Group raw = MPI_GROUP_EMPTY;
    MPI_Comm_group(MPI_COMM_WORLD, &raw);
    group     g   = group::from_native(raw);
    group_view gv = g; // implicit conversion
    EXPECT_EQ(gv.mpi_handle(), raw);
}

// ── group_accessors ───────────────────────────────────────────────────────────

TEST(GroupAccessorsTest, SizeMatchesCommSize) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_EQ(g.size(), world.size());
}

TEST(GroupAccessorsTest, RankMatchesCommRank) {
    comm_view world(MPI_COMM_WORLD);
    group     g   = world.group();
    auto      rnk = g.rank();
    ASSERT_TRUE(rnk.has_value());
    EXPECT_EQ(*rnk, world.rank());
}

TEST(GroupAccessorsTest, ContainsSelfTrue) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_TRUE(g.contains_self());
}

TEST(GroupAccessorsTest, CompareWorldWithItself) {
    comm_view world(MPI_COMM_WORLD);
    group     a = world.group();
    group     b = world.group();
    EXPECT_EQ(a.compare(b), GroupEquality::Identical);
}

// ── comm (owning) ─────────────────────────────────────────────────────────────

TEST(CommTest, ConceptSatisfied) {
    static_assert(convertible_to_mpi_handle<comm, MPI_Comm>);
}

TEST(CommTest, DupPreservesRankAndSize) {
    comm_view world(MPI_COMM_WORLD);

    // Free function: default-constructed comm as out-parameter
    comm dup_comm;
    dup(world, dup_comm);
    EXPECT_EQ(dup_comm.rank(), world.rank());
    EXPECT_EQ(dup_comm.size(), world.size());
    EXPECT_NE(dup_comm.mpi_handle(), MPI_COMM_NULL);
    EXPECT_NE(dup_comm.mpi_handle(), MPI_COMM_WORLD);

    // Member dup() on an owning comm
    comm dup2 = dup_comm.dup();
    EXPECT_EQ(dup2.rank(), world.rank());
    EXPECT_EQ(dup2.size(), world.size());
}

TEST(CommTest, MoveConstructTransfersOwnership) {
    comm_view world(MPI_COMM_WORLD);
    comm      a;
    dup(world, a);
    MPI_Comm raw = a.mpi_handle();
    comm     b   = std::move(a);

    EXPECT_EQ(a.mpi_handle(), MPI_COMM_NULL);
    EXPECT_EQ(b.mpi_handle(), raw);
}

TEST(CommTest, SplitEvenOdd) {
    comm_view world(MPI_COMM_WORLD);
    int       my_rank = world.rank();
    int       color   = my_rank % 2; // 0 = even, 1 = odd
    comm      sub;
    split(world, color, my_rank, sub);

    int world_size = world.size();
    int even_size  = (world_size + 1) / 2;
    int odd_size   = world_size / 2;
    int expected   = (color == 0) ? even_size : odd_size;

    EXPECT_EQ(sub.size(), expected);
    EXPECT_NE(sub.mpi_handle(), MPI_COMM_NULL);
}

TEST(CommTest, GroupFromCommView) {
    comm_view world(MPI_COMM_WORLD);
    group     g = world.group();
    EXPECT_EQ(g.size(), world.size());
}

TEST(CommTest, CommFromGroup) {
    comm_view world(MPI_COMM_WORLD);
    group     g        = world.group();
    comm      from_grp = comm(g, "test-tag");

    EXPECT_EQ(from_grp.rank(), world.rank());
    EXPECT_EQ(from_grp.size(), world.size());
}

TEST(CommTest, FromNative) {
    MPI_Comm raw = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &raw);
    comm c = comm::from_native(raw);
    EXPECT_EQ(c.mpi_handle(), raw);
    // freed by c's destructor
}
