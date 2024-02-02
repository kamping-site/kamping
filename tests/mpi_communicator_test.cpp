// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "test_assertions.hpp"

#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/comm_helper/num_numa_nodes.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

struct CommunicatorTest : Test {
    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int  flag;
        int* value;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
        EXPECT_TRUE(flag);
        mpi_tag_ub = *value;
    }

    int rank;
    int size;
    int mpi_tag_ub;
};

TEST_F(CommunicatorTest, empty_constructor) {
    Communicator comm;

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_WORLD);
    EXPECT_EQ(comm.rank(), rank);
    EXPECT_EQ(comm.rank_signed(), rank);
    EXPECT_EQ(comm.size_signed(), size);
    EXPECT_EQ(comm.size(), size);
    EXPECT_EQ(comm.root(), 0);
    EXPECT_EQ(comm.root_signed(), 0);
}

TEST_F(CommunicatorTest, constructor_with_mpi_communicator) {
    Communicator comm(MPI_COMM_SELF);

    int self_rank;
    int self_size;

    MPI_Comm_size(MPI_COMM_SELF, &self_size);
    MPI_Comm_rank(MPI_COMM_SELF, &self_rank);

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_SELF);
    EXPECT_EQ(comm.rank_signed(), self_rank);
    EXPECT_EQ(comm.rank(), self_rank);
    EXPECT_EQ(comm.size_signed(), self_size);
    EXPECT_EQ(comm.size(), self_size);
    EXPECT_EQ(comm.rank_signed(), 0);
    EXPECT_EQ(comm.rank(), 0);

    EXPECT_THROW(Communicator(MPI_COMM_NULL), kassert::KassertException);
}

TEST_F(CommunicatorTest, constructor_with_mpi_communicator_and_root) {
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(Communicator(MPI_COMM_WORLD, i), kassert::KassertException);
            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), kassert::KassertException);
        } else {
            Communicator comm(MPI_COMM_WORLD, i);
            ASSERT_EQ(comm.root(), i);

            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), kassert::KassertException);
        }
    }
}

TEST_F(CommunicatorTest, is_root) {
    Communicator comm;
    if (comm.root() == comm.rank()) {
        EXPECT_TRUE(comm.is_root());
    } else {
        EXPECT_FALSE(comm.is_root());
    }

    int const custom_root = comm.size_signed() - 1;
    if (custom_root == comm.rank_signed()) {
        EXPECT_TRUE(comm.is_root(custom_root));
    } else {
        EXPECT_FALSE(comm.is_root(custom_root));
    }
}

uint32_t mpi_abort_call_count           = 0;
int      mpi_abort_expected_return_code = 1;
MPI_Comm mpi_abort_expected_comm        = MPI_COMM_NULL;

int MPI_Abort(MPI_Comm comm, int errorcode) {
    mpi_abort_call_count++;
    EXPECT_EQ(errorcode, mpi_abort_expected_return_code);
    EXPECT_EQ(comm, mpi_abort_expected_comm);
    return 0;
}

TEST_F(CommunicatorTest, abort) {
    Communicator comm;

    mpi_abort_call_count           = 0;
    mpi_abort_expected_return_code = 1;
    mpi_abort_expected_comm        = comm.mpi_communicator();
    comm.abort();
    EXPECT_EQ(mpi_abort_call_count, 1);

    auto const new_comm = comm.split(0);

    mpi_abort_expected_return_code = 2;
    mpi_abort_expected_comm        = new_comm.mpi_communicator();
    new_comm.abort(2);
    EXPECT_EQ(mpi_abort_call_count, 2);
}

TEST_F(CommunicatorTest, set_root_bound_check) {
    Communicator comm;
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(comm.root(i), kassert::KassertException);
        } else {
            comm.root(i);
            EXPECT_EQ(i, comm.root());
            if (i > 0) {
                comm.root(asserting_cast<size_t>(i));
                EXPECT_EQ(i, comm.root());
            }
            if (comm.rank_signed() == i) {
                EXPECT_TRUE(comm.is_root());
            } else {
                EXPECT_FALSE(comm.is_root());
            }
        }
    }
}

TEST_F(CommunicatorTest, set_default_tag) {
    Communicator comm;
    ASSERT_EQ(comm.default_tag(), 0);
    comm.default_tag(1);
    ASSERT_EQ(comm.default_tag(), 1);
    comm.default_tag(23);
    ASSERT_EQ(comm.default_tag(), 23);
    comm.default_tag(mpi_tag_ub);
    ASSERT_EQ(comm.default_tag(), mpi_tag_ub);
    // Avoid signed integer overflow
    if (mpi_tag_ub < std::numeric_limits<decltype(mpi_tag_ub)>::max()) {
        EXPECT_THROW(comm.default_tag(mpi_tag_ub + 1), kassert::KassertException);
    }
    EXPECT_THROW(comm.default_tag(-1), kassert::KassertException);
}

TEST_F(CommunicatorTest, rank_shifted_checked) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i + rank < 0 || i + rank >= size) {
            EXPECT_THROW(((void)comm.rank_shifted_checked(i)), kassert::KassertException);
        } else {
            EXPECT_EQ(rank + i, comm.rank_shifted_checked(i));
        }
    }
}

TEST_F(CommunicatorTest, rank_shifted_cyclic) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        EXPECT_EQ((rank + i + 2 * size) % size, comm.rank_shifted_cyclic(i));
    }
}

TEST_F(CommunicatorTest, valid_rank) {
    Communicator comm;

    int mpi_size;
    MPI_Comm_size(comm.mpi_communicator(), &mpi_size);

    for (int i = -(2 * mpi_size); i < (2 * mpi_size); ++i) {
        EXPECT_EQ((i >= 0 && i < mpi_size), comm.is_valid_rank(i));
    }

    for (size_t i = 0; i < (2 * asserting_cast<size_t>(mpi_size)); ++i) {
        EXPECT_EQ(i < asserting_cast<size_t>(mpi_size), comm.is_valid_rank(i));
    }
}

TEST_F(CommunicatorTest, split_and_rank_conversion) {
    Communicator comm;

    // Test split with any number of reasonable colors
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);
        EXPECT_EQ(splitted_comm.size_signed(), expected_size);

        // Check for all rank ids whether they correctly convert to the splitted communicator
        for (int rank_to_test = 0; rank_to_test < size; ++rank_to_test) {
            int const expected_rank_in_splitted_comm = rank_to_test % i == color ? rank_to_test / i : MPI_UNDEFINED;
            EXPECT_EQ(expected_rank_in_splitted_comm, comm.convert_rank_to_communicator(rank_to_test, splitted_comm));
            EXPECT_EQ(expected_rank_in_splitted_comm, splitted_comm.convert_rank_from_communicator(rank_to_test, comm));
            if (expected_rank_in_splitted_comm != MPI_UNDEFINED) {
                EXPECT_EQ(
                    rank_to_test,
                    comm.convert_rank_from_communicator(expected_rank_in_splitted_comm, splitted_comm)
                );
                EXPECT_EQ(
                    rank_to_test,
                    splitted_comm.convert_rank_to_communicator(expected_rank_in_splitted_comm, comm)
                );
            }
        }
    }

    // Test split with any number of reasonable colors and inverse keys
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color, size - rank);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);
        EXPECT_EQ(splitted_comm.size_signed(), expected_size);

        int const smaller_ranks_in_split = rank / i;
        int const expected_rank          = expected_size - smaller_ranks_in_split - 1;
        EXPECT_EQ(splitted_comm.rank(), expected_rank);

        // Check for all rank ids whether they correctly convert to the splitted communicator
        for (int rank_to_test = 0; rank_to_test < size; ++rank_to_test) {
            int const expected_rank_rn_splitted_comm =
                rank_to_test % i == color ? expected_size - (rank_to_test / i) - 1 : MPI_UNDEFINED;
            EXPECT_EQ(expected_rank_rn_splitted_comm, comm.convert_rank_to_communicator(rank_to_test, splitted_comm));
            EXPECT_EQ(expected_rank_rn_splitted_comm, splitted_comm.convert_rank_from_communicator(rank_to_test, comm));
            if (expected_rank_rn_splitted_comm != MPI_UNDEFINED) {
                EXPECT_EQ(
                    rank_to_test,
                    comm.convert_rank_from_communicator(expected_rank_rn_splitted_comm, splitted_comm)
                );
                EXPECT_EQ(
                    rank_to_test,
                    splitted_comm.convert_rank_to_communicator(expected_rank_rn_splitted_comm, comm)
                );
            }
        }
    }
}

int      mpi_comm_split_type_expected_key;
MPI_Comm mpi_comm_split_type_expected_comm;
uint32_t mpi_comm_split_type_call_counter = 0;

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm* newcomm) {
    mpi_comm_split_type_call_counter++;
    EXPECT_EQ(mpi_comm_split_type_expected_comm, comm);
    EXPECT_EQ(mpi_comm_split_type_expected_key, key);
    EXPECT_EQ(info, MPI_INFO_NULL);
    EXPECT_EQ(split_type, MPI_COMM_TYPE_SHARED);
    return PMPI_Comm_split_type(comm, split_type, key, info, newcomm);
}

TEST_F(CommunicatorTest, split_by_type) {
    Communicator comm;

    // For this tests, we're assuming that we're running on a system in which each NUMA node has the same number of
    // ranks.
    mpi_comm_split_type_expected_key  = comm.rank_signed();
    mpi_comm_split_type_expected_comm = comm.mpi_communicator();
    ASSERT_GT(comm.num_numa_nodes(), 0);
    auto const shared_mem_comm_1 = comm.split_to_shared_memory();
    EXPECT_EQ(shared_mem_comm_1.size(), comm.size() / comm.num_numa_nodes());
    EXPECT_EQ(shared_mem_comm_1.rank(), comm.rank() % shared_mem_comm_1.size());

    mpi_comm_split_type_call_counter = 0;
    auto const shared_mem_comm_2     = comm.split_by_type(MPI_COMM_TYPE_SHARED);
    MPI_Group  group_1, group_2;
    MPI_Comm_group(shared_mem_comm_1.mpi_communicator(), &group_1);
    MPI_Comm_group(shared_mem_comm_2.mpi_communicator(), &group_2);
    int cmp;
    MPI_Group_compare(group_1, group_2, &cmp);
    EXPECT_EQ(cmp, MPI_IDENT);
    EXPECT_EQ(mpi_comm_split_type_call_counter, 1);

#ifdef OMPI_COMM_TYPE_L1CACHE
    constexpr size_t ranks_per_l1_cache = 1; // on all modern processors, assuming no oversubscription
    auto const       l1cache_comm       = comm.split_by_type(OMPI_COMM_TYPE_L1CACHE);
    EXPECT_EQ(l1cache_comm.size(), ranks_per_l1_cache);
#endif // OMPI_COMM_TYPE_L1CACHE

    mpi_comm_split_type_call_counter     = 0;
    mpi_comm_split_type_expected_key     = comm.rank_signed();
    mpi_comm_split_type_expected_comm    = comm.mpi_communicator();
    [[maybe_unused]] auto const new_comm = comm.split_to_shared_memory();
    EXPECT_EQ(mpi_comm_split_type_call_counter, 1);
}

TEST_F(CommunicatorTest, processor_name) {
    Communicator comm;

    char name[MPI_MAX_PROCESSOR_NAME];
    int  len;
    MPI_Get_processor_name(name, &len);

    ASSERT_EQ(comm.processor_name(), std::string(name, asserting_cast<size_t>(len)));
}

TEST_F(CommunicatorTest, create_communicators_via_provided_ranks) {
    Communicator comm;

    // Test communicator creation with any number of reasonable groups
    for (int i = 2; i <= size; ++i) {
        int const color = rank % i;
        // enumerate all ranks that are part of rank's new subcommunicator
        std::vector<int> ranks_in_own_group;
        for (int cur_rank = 0; cur_rank < size; ++cur_rank) {
            if (color == cur_rank % i) {
                ranks_in_own_group.push_back(cur_rank);
            }
        }
        auto subcommunicator          = comm.create_subcommunicators(ranks_in_own_group);
        auto expected_subcommunicator = comm.split(color);
        EXPECT_EQ(CommunicatorComparisonResult::congruent, subcommunicator.compare(expected_subcommunicator));
    }
}

TEST_F(CommunicatorTest, communicator_comparison) {
    Communicator comm;
    Communicator same_ranks_same_order = comm;
    // reverse rank order via key argument in split() method
    auto same_ranks_different_order = comm.split(0, size - rank);
    auto different_communicator     = comm.split(rank % 2);

    EXPECT_EQ(CommunicatorComparisonResult::identical, comm.compare(comm));
    EXPECT_EQ(CommunicatorComparisonResult::congruent, comm.compare(same_ranks_same_order));
    if (size > 1) {
        EXPECT_EQ(CommunicatorComparisonResult::similar, comm.compare(same_ranks_different_order));
        EXPECT_EQ(CommunicatorComparisonResult::unequal, comm.compare(different_communicator));
    }

    // test commutative property of communicator comparison
    EXPECT_EQ(CommunicatorComparisonResult::congruent, same_ranks_same_order.compare(comm));
    if (size > 1) {
        EXPECT_EQ(CommunicatorComparisonResult::similar, same_ranks_different_order.compare(comm));
        EXPECT_EQ(CommunicatorComparisonResult::unequal, different_communicator.compare(comm));
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(CommunicatorTest, create_communicators_via_provided_ranks_illegal_arguments) {
    Communicator comm;

    // set of ranks is empty
    EXPECT_KASSERT_FAILS(
        std::ignore = comm.create_subcommunicators(std::vector<int>{}),
        "The set of ranks to include in the new subcommunicator must not be empty."
    );
    // set of ranks must contain own rank
    EXPECT_KASSERT_FAILS(
        std::ignore = comm.create_subcommunicators(std::vector<int>{rank + 1}),
        "The ranks to include in the new subcommunicator must contain own rank."
    );
}
#endif

TEST_F(CommunicatorTest, create_communicators_via_provided_ranks_with_sparse_representation) {
    Communicator comm;
    // subcommunicator contains whole original communicator
    {
        std::vector<RankRange> rank_ranges{RankRange{0, size - 1, 1}};
        auto                   subcommunicator = comm.create_subcommunicators(RankRanges{rank_ranges});
        EXPECT_EQ(CommunicatorComparisonResult::congruent, subcommunicator.compare(comm));
    }
    // two subcommunicators (odd/even ranks)
    {
        if (size > 1) {
            int        last_odd_rank  = (size - 1) % 2 == 0 ? size - 2 : size - 1;
            int        last_even_rank = (size - 1) % 2 == 0 ? size - 1 : size - 2;
            RankRange  even_rank_range{0, last_even_rank, 2};
            RankRange  odd_rank_range{1, last_odd_rank, 2};
            bool const is_rank_even = rank % 2 == 0;
            RankRanges rank_ranges{std::vector<RankRange>{is_rank_even ? even_rank_range : odd_rank_range}};
            auto       subcommunicator          = comm.create_subcommunicators(rank_ranges);
            auto       expected_subcommunicator = comm.split(is_rank_even);
            EXPECT_EQ(CommunicatorComparisonResult::congruent, subcommunicator.compare(expected_subcommunicator));
        }
    }
    // two ranges spanning whole communicator
    {
        if (size > 1) {
            RankRange  first_half{0, (size / 2) - 1, 1};
            RankRange  second_half{size / 2, size - 1, 1};
            RankRanges rank_ranges{std::vector<RankRange>{first_half, second_half}};
            auto       subcommunicator = comm.create_subcommunicators(rank_ranges);
            EXPECT_EQ(CommunicatorComparisonResult::congruent, subcommunicator.compare(comm));
        }
        if (size > 1) {
            int        last_odd_rank  = (size - 1) % 2 == 0 ? size - 2 : size - 1;
            int        last_even_rank = (size - 1) % 2 == 0 ? size - 1 : size - 2;
            RankRange  even_rank_range{0, last_even_rank, 2};
            RankRange  odd_rank_range{1, last_odd_rank, 2};
            RankRanges rank_ranges{std::vector<RankRange>{even_rank_range, odd_rank_range}};
            auto       subcommunicator = comm.create_subcommunicators(rank_ranges);
            EXPECT_EQ(
                CommunicatorComparisonResult::similar,
                subcommunicator.compare(comm)
            ); // communicators are not congruent as the rank order differs
        }
    }
}

// Using KAMPING_ASSERTION_LEVEL_HEAVY because EXPECT_KASSERT_FAILS only checks for failing kasserts on that level
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_HEAVY)
TEST_F(CommunicatorTest, create_communicators_via_provided_ranks_with_sparse_representation_illegal_arguments) {
    Communicator comm;

    // set of ranks is empty
    EXPECT_KASSERT_FAILS(
        std::ignore = comm.create_subcommunicators(RankRanges(nullptr, 0)),
        "The set of ranks to include in the new subcommunicator must not be empty."
    );
    EXPECT_KASSERT_FAILS(
        std::ignore = comm.create_subcommunicators(RankRanges(std::vector<RankRange>{})),
        "The set of ranks to include in the new subcommunicator must not be empty."
    );
    // set of ranks must contain own rank
    if (size > 1) {
        int rank_range_array[1][3] = {{size, size + 1, 1}};
        EXPECT_KASSERT_FAILS(
            std::ignore = comm.create_subcommunicators(RankRanges(rank_range_array, 1)),
            "The ranks to include in the new subcommunicator must contain own rank."
        );
        EXPECT_KASSERT_FAILS(
            std::ignore =
                comm.create_subcommunicators(RankRanges(std::vector<RankRange>{RankRange{size, size + 1, 1}})),
            "The ranks to include in the new subcommunicator must contain own rank."
        );
    }
}
#endif

TEST_F(CommunicatorTest, assignment) {
    // move assignment
    Communicator comm;
    comm = Communicator();

    // copy assignment
    Communicator comm2;
    comm = comm2;
}

TEST_F(CommunicatorTest, comm_world) {
    // These are what comm_world is intended for.
    EXPECT_EQ(comm_world().rank(), rank);
    EXPECT_EQ(comm_world().size(), size);
    EXPECT_EQ(comm_world().rank_signed(), rank);
    EXPECT_EQ(comm_world().size_signed(), size);
}

TEST_F(CommunicatorTest, comm_world_convenience_functions) {
    EXPECT_EQ(world_rank(), rank);
    EXPECT_EQ(world_size(), size);
    EXPECT_EQ(world_rank_signed(), rank);
    EXPECT_EQ(world_size_signed(), size);
}

TEST_F(CommunicatorTest, swap) {
    BasicCommunicator comm1;
    MPI_Comm          mpi_comm1  = comm1.mpi_communicator();
    auto              root_comm1 = 1 % size;
    comm1.root(root_comm1);
    comm1.default_tag(1);

    int const color      = rank % 2;
    auto      comm2      = comm1.split(color);
    MPI_Comm  mpi_comm2  = comm2.mpi_communicator();
    auto      size_comm2 = comm2.size();
    auto      rank_comm2 = comm2.rank();
    auto      root_comm2 = 2 % size_comm2;
    comm2.root(root_comm2);
    comm2.default_tag(2);

    EXPECT_NE(mpi_comm1, mpi_comm2);

    comm1.swap(comm2);

    EXPECT_EQ(comm1.mpi_communicator(), mpi_comm2);
    EXPECT_EQ(comm1.size(), size_comm2);
    EXPECT_EQ(comm1.rank(), rank_comm2);
    EXPECT_EQ(comm1.root(), root_comm2);
    EXPECT_EQ(comm1.default_tag(), 2);

    EXPECT_EQ(comm2.mpi_communicator(), mpi_comm1);
    EXPECT_EQ(comm2.size(), size);
    EXPECT_EQ(comm2.rank(), rank);
    EXPECT_EQ(comm2.root(), root_comm1);
    EXPECT_EQ(comm2.default_tag(), 1);
}

static std::vector<MPI_Comm> freed_communicators;
static bool                  track_freed_communicators = false;

template <typename T>
bool vector_contains(std::vector<T> const& vec, T const& elem) {
    return std::find(vec.begin(), vec.end(), elem) != vec.end();
}

bool was_freed(MPI_Comm const& comm) {
    return vector_contains(freed_communicators, comm);
}

int MPI_Comm_free(MPI_Comm* comm) {
    if (track_freed_communicators) {
        EXPECT_FALSE(vector_contains(freed_communicators, *comm));
        freed_communicators.push_back(*comm);
    }
    return PMPI_Comm_free(comm);
}

TEST_F(CommunicatorTest, communicator_management) {
    track_freed_communicators = true;
    MPI_Comm user_owned_mpi_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &user_owned_mpi_comm);
    MPI_Comm lib_owned_mpi_comm = MPI_COMM_NULL;

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Base functionality with ownership set in constructor.
    {
        BasicCommunicator non_owning_comm1(user_owned_mpi_comm, false);
        // Default should be non-owning.
        BasicCommunicator non_owning_comm2(user_owned_mpi_comm);
        MPI_Comm_dup(MPI_COMM_WORLD, &lib_owned_mpi_comm);
        BasicCommunicator owning_comm(lib_owned_mpi_comm, true);
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Default constructed communicator should not be free..
    MPI_Comm comm_world;
    {
        BasicCommunicator owning_comm;
        comm_world = owning_comm.mpi_communicator();
    }
    EXPECT_FALSE(was_freed(comm_world));
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Split.
    {
        BasicCommunicator non_owning_comm(user_owned_mpi_comm, false);
        int const         color = 0;
        // Splitting should create an owned MPI_Comm.
        BasicCommunicator owning_comm = non_owning_comm.split(color);
        lib_owned_mpi_comm            = owning_comm.mpi_communicator();
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Copy assignment.
    {
        BasicCommunicator non_owning_comm(user_owned_mpi_comm, false);
        // Copy assignment should create an owned MPI_Comm.
        BasicCommunicator owning_comm = non_owning_comm;
        EXPECT_NE(owning_comm.mpi_communicator(), non_owning_comm.mpi_communicator());
        lib_owned_mpi_comm = owning_comm.mpi_communicator();
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Copy constructor.
    {
        BasicCommunicator non_owning_comm(user_owned_mpi_comm, false);
        // Copy construction should create an owned MPI_Comm.
        BasicCommunicator owning_comm(non_owning_comm);
        EXPECT_NE(owning_comm.mpi_communicator(), non_owning_comm.mpi_communicator());
        lib_owned_mpi_comm = owning_comm.mpi_communicator();
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Move constructor.
    {
        BasicCommunicator non_owning_comm1(user_owned_mpi_comm, false);
        // Move construction should not change ownership of MPI_Comms.
        BasicCommunicator non_owning_comm2(std::move(non_owning_comm1));
    }
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Move assignment.
    {
        BasicCommunicator non_owning_comm(user_owned_mpi_comm, false);
        MPI_Comm_dup(MPI_COMM_WORLD, &lib_owned_mpi_comm);
        BasicCommunicator comm2(lib_owned_mpi_comm, true);
        // Move assignment should not change ownership of MPI_Comms.
        comm2 = std::move(non_owning_comm);
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Move constructing should not cause a communicator to be freed twice.
    {
        MPI_Comm_dup(MPI_COMM_WORLD, &lib_owned_mpi_comm);
        BasicCommunicator owning_comm1(lib_owned_mpi_comm, true);
        BasicCommunicator owning_comm2(std::move(owning_comm1));
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Move assignment should not cause a communicator to be freed twice.
    {
        MPI_Comm_dup(MPI_COMM_WORLD, &lib_owned_mpi_comm);
        BasicCommunicator owning_comm1(lib_owned_mpi_comm, true);
        BasicCommunicator owning_comm2(user_owned_mpi_comm, false);
        owning_comm2 = std::move(owning_comm1);
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Disowning.
    {
        BasicCommunicator owning_comm(user_owned_mpi_comm, true);
        MPI_Comm          contained_comm = owning_comm.disown_mpi_communicator();
        EXPECT_EQ(user_owned_mpi_comm, contained_comm);
    }
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Swapping.
    {
        BasicCommunicator comm1(user_owned_mpi_comm, false);
        MPI_Comm_dup(MPI_COMM_WORLD, &lib_owned_mpi_comm);
        BasicCommunicator comm2(lib_owned_mpi_comm, true);
        // Swapping should not change ownership of MPI_Comms.
        comm1.swap(comm2);
    }
    EXPECT_TRUE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    // Cleanly free the communicator.
    MPI_Comm_free(&user_owned_mpi_comm);

    // Reset list of freed communicators
    freed_communicators.clear();
    EXPECT_FALSE(was_freed(lib_owned_mpi_comm));
    EXPECT_FALSE(was_freed(user_owned_mpi_comm));

    track_freed_communicators = false;
}
