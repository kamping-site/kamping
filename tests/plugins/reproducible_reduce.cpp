#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "kamping/collectives/scatter.hpp"
#include "kamping/plugin/reproducible_reduce.hpp"
#include "kamping/communicator.hpp"

using namespace kamping::plugin::reproducible_reduce;


// Reduction tree with 7 indices to further clarify the test cases below
//
// │                  
// ├───────────┐      
// │           │      
// ├─────┐     ├─────┐
// │     │     │     │
// ├──┐  ├──┐  ├──┐  │
// │  │  │  │  │  │  │
// 0  1  2  3  4  5  6
//          +--------+ region 1
//    +-----+          region 2
// +-----------+       region 3
//
// |----|-|-----------  distribution
//    1  0      2       rank
TEST(ReproducibleReduceTest, TreeParent) {
    EXPECT_EQ(0, tree_parent(2));
    EXPECT_EQ(0, tree_parent(4));
    EXPECT_EQ(4, tree_parent(5));
    EXPECT_EQ(0, tree_parent(4));
    EXPECT_EQ(4, tree_parent(6));
}

TEST(ReproducibleReduceTest, TreeSubtreeSize) {
    EXPECT_EQ(2, tree_subtree_size(2));
    EXPECT_EQ(1, tree_subtree_size(3));
    EXPECT_EQ(4, tree_subtree_size(4));
}

TEST(ReproducibleReduceTest, TreeRankIntersection) {
    // region 1

    EXPECT_THAT(
            tree_rank_intersecting_elements(3, 6),
            ::testing::ElementsAre(3, 4));

    // region 2
    EXPECT_THAT(
            tree_rank_intersecting_elements(1, 3),
            ::testing::ElementsAre(1, 2));

    // region 3
    EXPECT_THAT(
            tree_rank_intersecting_elements(0, 4),
            ::testing::IsEmpty());
}

TEST(ReproducibleReduceTest, TreeRankCalculation) {
    // See introductory comment for visualization of range
    std::map<size_t, size_t> start_indices {{0, 1}, {2, 0}, {3, 2}, {7,3}};
    
    auto calc_rank = [&start_indices](auto i) { return tree_rank_from_index_map(start_indices, i); };

    EXPECT_EQ(1, calc_rank(0U));
    EXPECT_EQ(1, calc_rank(1U));
    EXPECT_EQ(0, calc_rank(2U));
    EXPECT_EQ(2, calc_rank(3U));
    EXPECT_EQ(2, calc_rank(4U));
    EXPECT_EQ(2, calc_rank(5U));
    EXPECT_EQ(2, calc_rank(6U));

    for (auto i = 7UL; i < 80000; ++i) {
        EXPECT_EQ(3, calc_rank(i));
    }

    // TODO: add test for edge cases (empty index map)
}


void attach_debugger(bool);

TEST(ReproducibleReduceTest, PluginInit) {
    double const        epsilon = std::numeric_limits<double>::epsilon();
    std::vector<double> test_array{1, 1 + epsilon, 2 + epsilon, epsilon, 8, 9};

    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;
    if (const auto debug_rank = getenv("DEBUG_MPI_RANK"); debug_rank != nullptr) {
        attach_debugger(comm.rank() == std::atoi(debug_rank));
    }


    int              values_per_rank = test_array.size() / comm.size();
    std::vector<int> send_counts(comm.size(), values_per_rank);
    std::vector<int> recv_displs;

    size_t start_index = 0;
    for (int i = 0; i < kamping::asserting_cast<int>(comm.size()); i++) {
        recv_displs.push_back(start_index);
        start_index += send_counts[i];
    }
    ASSERT_EQ(recv_displs.size(), comm.size());

    // Distribute test array to individual ranks
    std::vector<double> local_array;
    comm.scatterv(
            kamping::send_buf(test_array),
            kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(local_array),
            kamping::send_counts(send_counts),
            kamping::send_displs(recv_displs)
    );

    printf("Rank %li, arr = {", comm.rank());
    for (auto v : local_array) {
        printf("%f, ", v);
    }
    printf("}\n");


    auto reproducible_comm =
        comm.make_reproducible_comm<double>(kamping::recv_displs(recv_displs), kamping::send_counts(send_counts));

    const auto reference_val = std::reduce(test_array.begin(), test_array.end(), 0.0, std::plus<>());
    std::cout << "Reference sum: " << reference_val << "\n";
    auto v = reproducible_comm.reproducible_reduce(
            kamping::send_buf(local_array),
            kamping::op(kamping::ops::plus<double>{}));
    std::cout << "Computed sum: " << v << "\n";

    EXPECT_EQ(v, reference_val);
}

#include <fstream>
#include <unistd.h>
void __attribute__((optimize("O0"))) attach_debugger(bool condition) {
    if (!condition) return;
    volatile bool attached = false;

    // also write PID to a file
    std::ofstream os("/tmp/mpi_debug.pid");
    os << getpid() << "\n";
    os.close();

    std::cout << "Waiting for debugger to be attached, PID: "
        << getpid() << "\n";
    while (!attached) sleep(1);
}
