#include "../test_assertions.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/reproducible_reduce.hpp"

using namespace kamping::plugin::reproducible_reduce;

using Distribution = struct Distribution {
    std::vector<int> send_counts;
    std::vector<int> displs;

    Distribution(std::vector<int> _send_counts, std::vector<int> recv_displs)
        : send_counts(_send_counts),
          displs(recv_displs) {}
};

template <typename C, typename T>
std::vector<T> scatter_array(C comm, std::vector<T> const& global_array, Distribution const d) {
    std::vector<T> result;

    comm.scatterv(
        kamping::send_buf(global_array),
        kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result),
        kamping::send_counts(d.send_counts),
        kamping::send_displs(d.displs)
    );

    return result;
}

auto displacement_from_sendcounts(std::vector<int>& send_counts) {
    std::vector<int> displacement;
    displacement.reserve(send_counts.size());

    int start_index = 0;
    for (auto const& send_count: send_counts) {
        displacement.push_back(start_index);
        start_index += send_count;
    }

    return displacement;
}

auto distribute_evenly(size_t const collection_size, size_t const comm_size) {
    auto const elements_per_rank = collection_size / comm_size;
    auto const remainder         = collection_size % comm_size;

    std::vector<int> send_counts(comm_size, kamping::asserting_cast<int>(elements_per_rank));
    std::for_each_n(send_counts.begin(), remainder, [](auto& n) { n += 1; });

    return Distribution(send_counts, displacement_from_sendcounts(send_counts));
}

auto distribute_randomly(size_t const collection_size, size_t const comm_size, size_t const seed) {
    std::mt19937                    rng(seed);
    std::uniform_int_distribution<> dist(0, kamping::asserting_cast<int>(collection_size));

    // See https://stackoverflow.com/a/48205426 for details
    std::vector<int> points(comm_size, 0UL);
    points.push_back(kamping::asserting_cast<int>(collection_size));
    std::generate(points.begin() + 1, points.end() - 1, [&dist, &rng]() { return dist(rng); });
    std::sort(points.begin(), points.end());

    std::vector<int> send_counts(comm_size);
    for (size_t i = 0; i < send_counts.size(); ++i) {
        send_counts[i] = points[i + 1] - points[i];
    }

    // Shuffle to generate distributions where start indices are not monotonically increasing
    std::vector<size_t> indices(send_counts.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    auto displacement = displacement_from_sendcounts(send_counts);
    EXPECT_EQ(send_counts.size(), displacement.size());

    decltype(send_counts)  shuffled_send_counts(send_counts.size(), 0);
    decltype(displacement) shuffled_displacement(displacement.size(), 0);
    for (auto i = 0UL; i < send_counts.size(); ++i) {
        shuffled_send_counts[i]  = send_counts[indices[i]];
        shuffled_displacement[i] = displacement[indices[i]];
    }

    EXPECT_EQ(
        collection_size,
        std::reduce(shuffled_send_counts.begin(), shuffled_send_counts.end(), 0UL, std::plus<>())
    );

    return Distribution(shuffled_send_counts, shuffled_displacement);
}
auto generate_test_vector(size_t length, size_t seed) {
    std::mt19937                   rng(seed);
    std::uniform_real_distribution distr;
    std::vector<double>            result(length);
    std::generate(result.begin(), result.end(), [&distr, &rng]() { return distr(rng); });

    return result;
}

// Test generators
TEST(ReproducibleReduceTest, DistributionGeneration) {
    Distribution distr1 = distribute_evenly(9, 4);
    EXPECT_THAT(distr1.send_counts, testing::ElementsAre(3, 2, 2, 2));
    EXPECT_THAT(distr1.displs, testing::ElementsAre(0, 3, 5, 7));

    Distribution distr2 = distribute_evenly(2, 5);
    EXPECT_THAT(distr2.send_counts, testing::ElementsAre(1, 1, 0, 0, 0));
    EXPECT_THAT(distr2.displs, testing::ElementsAre(0, 1, 2, 2, 2));

    Distribution distr3 = distribute_randomly(30, 4, 42);
    EXPECT_EQ(distr3.send_counts.size(), 4);
    EXPECT_THAT(std::accumulate(distr3.send_counts.begin(), distr3.send_counts.end(), 0), 30);
}

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
    EXPECT_THAT(tree_rank_intersecting_elements(3, 6), ::testing::ElementsAre(3, 4));

    // region 2
    EXPECT_THAT(tree_rank_intersecting_elements(1, 3), ::testing::ElementsAre(1, 2));

    // region 3
    EXPECT_THAT(tree_rank_intersecting_elements(0, 4), ::testing::IsEmpty());
}

TEST(ReproducibleReduceTest, TreeRankCalculation) {
    // See introductory comment for visualization of range
    std::map<size_t, size_t> start_indices{{0, 1}, {2, 0}, {3, 2}, {7, 3}};

    auto calc_rank = [&start_indices](auto i) {
        return tree_rank_from_index_map(start_indices, i);
    };

    EXPECT_EQ(1, calc_rank(0U));
    EXPECT_EQ(1, calc_rank(1U));
    EXPECT_EQ(0, calc_rank(2U));
    EXPECT_EQ(2, calc_rank(3U));
    EXPECT_EQ(2, calc_rank(4U));
    EXPECT_EQ(2, calc_rank(5U));
    EXPECT_EQ(2, calc_rank(6U));

    // TODO: add test for edge cases (empty index map)
}

TEST(ReproducibleReduceTest, Log2l) {
    using kamping::plugin::reproducible_reduce::log2l;

    EXPECT_EQ(log2l(1), 0);
    EXPECT_EQ(log2l(2), 1);
    EXPECT_EQ(log2l(3), 1);
    EXPECT_EQ(log2l(4), 2);
    EXPECT_EQ(log2l(5), 2);
    EXPECT_EQ(log2l(8), 3);
    EXPECT_EQ(log2l(9), 3);
    EXPECT_EQ(log2l(15), 3);
    EXPECT_EQ(log2l(16), 4);
    EXPECT_EQ(log2l(17), 4);
}

TEST(ReproducibleReduceTest, TreeLevelCalculation) {
    EXPECT_EQ(tree_height(5), 3); // Tree with 5 elements has 3 layers

    EXPECT_EQ(tree_height(0), 0);
    EXPECT_EQ(tree_height(1), 0);
    EXPECT_EQ(tree_height(2), 1);
    EXPECT_EQ(tree_height(3), 2);
    EXPECT_EQ(tree_height(4), 2);
    EXPECT_EQ(tree_height(5), 3);
    EXPECT_EQ(tree_height(15), 4);
    EXPECT_EQ(tree_height(16), 4);
    EXPECT_EQ(tree_height(17), 5);

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(subtree_height(0), "");
#endif

    // Randomized testing
    std::random_device rd;
    size_t             seed = rd();
    std::mt19937       rng(seed);

    std::uniform_int_distribution<size_t> size_distribution;

    auto checks = 0UL;
    for (auto i = 0UL; i < 50; ++i) {
        auto generated = size_distribution(rng);

        // Compare to expressions from previous implementation
        auto root_val = ceil(log2(static_cast<double>(generated)));
        EXPECT_EQ(root_val, tree_height(generated));

        if (generated != 0) {
            auto subtree_val = log2(static_cast<double>(tree_subtree_size(generated)));
            EXPECT_EQ(subtree_val, subtree_height(generated));
        }

        ++checks;
    }
}

constexpr double const epsilon = std::numeric_limits<double>::epsilon();
TEST(ReproducibleReduceTest, SimpleSum) {
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> full_comm;
    constexpr int                                                                 comm_size = 2;
    ASSERT_GE(full_comm.size(), comm_size) << "Comm is of insufficient size";
    auto comm = full_comm.split(full_comm.rank() < comm_size);

    if (full_comm.rank() >= comm_size)
        return;
    ASSERT_EQ(comm.size(), comm_size);

    std::vector const a{1e3, epsilon, epsilon / 2, epsilon / 2};
    EXPECT_EQ(std::accumulate(a.begin(), a.end(), 0.0), 1e3 + epsilon);

    Distribution distr({2, 2}, {0, 2});
    ASSERT_EQ(comm.size(), 2);

    auto local_a = scatter_array(comm, a, distr);

    ASSERT_EQ(comm.size(), distr.send_counts.size());
    ASSERT_EQ(comm.size(), distr.displs.size());
    auto repr_comm = comm.make_reproducible_comm<double>(
        kamping::send_counts(distr.send_counts),
        kamping::recv_displs(distr.displs)
    );

    double sum = repr_comm.reproducible_reduce(kamping::send_buf(local_a), kamping::op(kamping::ops::plus<>()));
    EXPECT_EQ(sum, (1e3 + epsilon) + (epsilon / 2 + epsilon / 2));
}

template <typename F>
void with_comm_size_n(
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> const& comm, size_t comm_size, F f
) {
    KASSERT(comm.is_same_on_all_ranks(comm_size), "Target comm_size must be same on all ranks");
    KASSERT(
        comm.size() >= comm_size,
        "Can not create communicator with " << comm_size << " ranks when process only has " << comm.size()
                                            << " ranks assigned."
    );

    int  rank_active = comm.rank() < comm_size;
    auto new_comm    = comm.split(rank_active);

    if (rank_active) {
        KASSERT(new_comm.size() == comm_size);
        f(new_comm);
    }
}

TEST(ReproducibleReduceTest, WorksWithNonzeroRoot) {
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> full_comm;
    ASSERT_GE(full_comm.size(), 2) << "Comm is of insufficient size";

    std::vector<double> array{1.0, 2.0, 3.0, 4.0};
    Distribution        distribution({0, 4}, {0, 0});

    with_comm_size_n(full_comm, 2, [&distribution, &array](auto comm) {
        auto repr_comm = comm.template make_reproducible_comm<double>(
            kamping::send_counts(distribution.send_counts),
            kamping::recv_displs(distribution.displs)
        );

        auto local_array = scatter_array(comm, array, distribution);

        double result =
            repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(kamping::ops::plus<>{}));

        EXPECT_EQ(result, (1.0 + 2.0) + (3.0 + 4.0));
    });
}

TEST(ReproducibleReduceTest, Fuzzing) {
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;
    comm.barrier();

    ASSERT_GT(comm.size(), 1) << "Fuzzing with only one rank is useless";

    constexpr auto NUM_ARRAYS        = 2; // 15;
    constexpr auto NUM_DISTRIBUTIONS = 3; // 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long      seed;
    if (comm.is_root()) {
        seed = rd();
    }
    comm.bcast_single(kamping::send_recv_buf(seed));

    std::uniform_int_distribution<size_t> array_length_distribution(1, 20);
    std::uniform_int_distribution<size_t> rank_distribution(1, comm.size());
    std::mt19937                          rng(seed);       // RNG for distribution & rank number
    std::mt19937                          rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const        data_array_size = array_length_distribution(rng);
        ASSERT_NE(0, data_array_size);
        if (comm.is_root()) {
            data_array = generate_test_vector(data_array_size, rng_root());
        }
        double reference_result = 0;

        // Calculate reference result
        with_comm_size_n(comm, 1, [&reference_result, &data_array](auto comm_) {
            KASSERT(comm_.size() == 1);
            const auto distribution = distribute_evenly(data_array.size(), 1);
            auto       repr_comm    = comm_.template make_reproducible_comm<double>(
                kamping::send_counts(distribution.send_counts),
                kamping::recv_displs(distribution.displs)
            );

            reference_result =
                repr_comm.reproducible_reduce(kamping::send_buf(data_array), kamping::op(kamping::ops::plus<>{}));

            // Sanity check
            ASSERT_NEAR(reference_result, std::accumulate(data_array.begin(), data_array.end(), 0.0), 1e-9);
        });

        comm.barrier();

        for (auto j = 0U; j < NUM_DISTRIBUTIONS; ++j) {
            auto const ranks        = rank_distribution(rng);
            auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

            with_comm_size_n(comm, ranks, [&distribution, &data_array, &reference_result, &checks, &ranks](auto comm_) {
                comm_.barrier();
                ASSERT_EQ(ranks, comm_.size());
                // Since not all ranks execute this function, rng may not be used to avoid it from falling out of sync

                auto repr_comm = comm_.template make_reproducible_comm<double>(
                    kamping::send_counts(distribution.send_counts),
                    kamping::recv_displs(distribution.displs)
                );

                std::vector<double> local_arr = scatter_array(comm_, data_array, distribution);

                double computed_result =
                    repr_comm.reproducible_reduce(kamping::send_buf(local_arr), kamping::op(kamping::ops::plus<>{}));

                if (comm_.is_root()) {
                    EXPECT_EQ(computed_result, reference_result);
                }
                ++checks;
            });
        }
    }

    if (comm.is_root()) {
    }
}

TEST(ReproducibleReduceTest, ReproducibleResults) {
    auto const                                                                    v_size = 50;
    auto const                                                                    v = generate_test_vector(v_size, 42);
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;

    double reference_result = 0.0;

    // Calculate reference
    with_comm_size_n(comm, 1, [&reference_result, v_size, &v](auto sub_comm) {
        auto repr_comm = sub_comm.template make_reproducible_comm<double>(
            kamping::send_counts({kamping::asserting_cast<int>(v_size)}),
            kamping::recv_displs({0})
        );
        reference_result =
            repr_comm.template reproducible_reduce(kamping::send_buf(v), kamping::op(kamping::ops::plus<double>{}));
    });

    comm.bcast_single(kamping::send_recv_buf(reference_result));

    for (auto i = 2U; i <= comm.size(); ++i) {
        with_comm_size_n(comm, i, [&v, i, reference_result](auto subcomm) {
            auto distr    = distribute_randomly(v.size(), i, 43 + i);
            auto reprcomm = subcomm.template make_reproducible_comm<double>(
                kamping::send_counts(distr.send_counts),
                kamping::recv_displs(distr.displs)
            );

            // Distribute global array across cluster
            std::vector<double> local_v = scatter_array(subcomm, v, distr);
            subcomm.scatterv(
                kamping::send_buf(v),
                kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(local_v),
                kamping::send_counts(distr.send_counts),
                kamping::send_displs(distr.displs)
            );

            double const result =
                reprcomm.reproducible_reduce(kamping::send_buf(local_v), kamping::op(kamping::ops::plus<double>{}));

            EXPECT_EQ(reference_result, result) << "Irreproducible result for p=" << i;
        });
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(ReproducibleReduceTest, ErrorChecking) {
    // Test error messages on communicator with 3 ranks
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;
    with_comm_size_n(comm, 3, [](auto sub_comm) {
        EXPECT_EQ(sub_comm.size(), 3);

        // Correct distribution
        EXPECT_NO_THROW(sub_comm.template make_reproducible_comm<double>(
            kamping::send_counts({5, 5, 5}),
            kamping::recv_displs({0, 5, 10})
        ));

        // Supplied distribution has unequal lengths
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5, 5}),
                kamping::recv_displs({0, 5, 10})
            ),
            ""
        );

        // Supplied distribution does not match communicator size
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5, 5}),
                kamping::recv_displs({0, 5, 10, 15})
            ),
            ""
        );

        // Supplied distribution does not start at 0
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5}),
                kamping::recv_displs({5, 10, 15})
            ),
            ""
        );

        // Supplied distribution has gaps
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5}),
                kamping::recv_displs({0, 10, 15})
            ),
            ""
        );

        // Supplied distribution has invalid displacements
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5}),
                kamping::recv_displs({0, 0, 0})
            ),
            ""
        );

        // Supplied distribution has negative displacement
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({5, 5, 5}),
                kamping::recv_displs({-5, 0, 5})
            ),
            ""
        );

        // Supplied distribution is empty
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts(std::vector<int>()),
                kamping::recv_displs(std::vector<int>())
            ),
            ""
        );

        // Empty array, send_counts all zero
        EXPECT_KASSERT_FAILS(
            sub_comm.template make_reproducible_comm<double>(
                kamping::send_counts({0, 0, 0}),
                kamping::recv_displs({0, 0, 0})
            ),
            ""
        );
    });
}
#endif

double multiply(double const& lhs, double const& rhs) {
    return lhs * rhs;
}

TEST(ReproducibleReduceTest, OtherOperations) {
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;

    std::vector<double> const array({5, 2, 3, 1, 7});

    size_t seed;
    if (comm.is_root()) {
        std::random_device rd;
        seed = rd();
    }
    comm.bcast_single(kamping::send_recv_buf(seed));

    std::mt19937 rng(seed); // RNG for distribution & rank number

    auto const distr = distribute_randomly(array.size(), comm.size(), rng());

    std::vector<double> local_array = scatter_array(comm, array, distr);

    auto repr_comm = comm.template make_reproducible_comm<double>(
        kamping::send_counts(distr.send_counts),
        kamping::recv_displs(distr.displs)
    );

    auto const max_val =
        repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(kamping::ops::max<>{}));
    EXPECT_EQ(max_val, 7.0);

    auto const min_val =
        repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(kamping::ops::min<>{}));
    EXPECT_EQ(min_val, 1.0);

    auto const product =
        repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(kamping::ops::multiplies<>{}));
    EXPECT_EQ(product, 5.0 * 2.0 * 3.0 * 1.0 * 7.0);

    // Use lambda
    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };
    auto result = repr_comm.reproducible_reduce(
        kamping::send_buf(local_array),
        kamping::op(std::move(add_plus_42_lambda), kamping::ops::commutative)
    );
    EXPECT_EQ(result, array[0] + array[1] + array[2] + array[3] + array[4] + 4 * 42);

    // Inline lambda
    auto subtracted = repr_comm.reproducible_reduce(
        kamping::send_buf(local_array),
        kamping::op([](auto const& lhs, auto const& rhs) { return lhs - rhs; }, kamping::ops::non_commutative)
    );
    EXPECT_EQ(subtracted, ((array[0] - array[1]) - (array[2] - array[3])) - array[4]);

    // function object
    struct Plus3 {
        double operator()(double const& lhs, double const& rhs) const {
            return lhs + rhs + 3.0;
        }
    };
    EXPECT_EQ(Plus3{}(1.0, 2.0), 1.0 + 2.0 + 3.0);
    auto plus3_result =
        repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(Plus3{}, kamping::ops::commutative));
    EXPECT_EQ(plus3_result, array[0] + array[1] + array[2] + array[3] + array[4] + 4 * 3);

    // function pointer
    auto multiply_ptr =
        repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(multiply, kamping::ops::commutative));
    EXPECT_EQ(multiply_ptr, product);
}

auto compute_mean_stddev(std::vector<double>& array) {
    auto const size = static_cast<double>(array.size());
    auto const mean = std::accumulate(array.begin(), array.end(), 0.0) / size;

    auto const variance =
        std::accumulate(array.begin(), array.end(), 0.0, [&mean, size](auto accumulator, auto const& v) {
            return (accumulator + ((v - mean) * (v - mean) / (static_cast<double>(size) - 1.0)));
        });

    return std::make_pair(mean, std::sqrt(variance));
}
TEST(ReproducibleReduceTest, Microbenchmark) {
    kamping::Communicator<std::vector, kamping::plugin::ReproducibleReducePlugin> comm;

    std::vector<double> array;
    constexpr auto      array_size  = 100U;
    constexpr auto      repetitions = 3;

    size_t seed;
    if (comm.is_root()) {
        std::random_device rd;
        seed  = rd();
        array = generate_test_vector(array_size, seed);
    }
    comm.bcast_single(kamping::send_recv_buf(seed));

    std::mt19937 rng(seed); // RNG for distribution & rank number

    std::vector<std::chrono::time_point<std::chrono::system_clock>> timings;
    timings.reserve(repetitions + 1);

    auto const          distr       = distribute_evenly(array_size, comm.size());
    std::vector<double> local_array = scatter_array(comm, array, distr);

    auto repr_comm = comm.template make_reproducible_comm<double>(
        kamping::send_counts(distr.send_counts),
        kamping::recv_displs(distr.displs)
    );

    double result;
    timings.push_back(std::chrono::system_clock::now());
    for (auto i = 0U; i < repetitions; ++i) {
        result = repr_comm.reproducible_reduce(kamping::send_buf(local_array), kamping::op(kamping::ops::plus<>{}));
        timings.push_back(std::chrono::system_clock::now());
    }

    if (comm.is_root()) {
        EXPECT_NEAR(result, std::accumulate(array.begin(), array.end(), 0.0), 1e-9);

        std::vector<double> iteration_time(repetitions);

        for (auto i = 0U; i < repetitions; ++i) {
            iteration_time[i] = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(timings[i + 1] - timings[i]).count()
            );
        }
        auto const r = compute_mean_stddev(iteration_time);
        RecordProperty("Mean", static_cast<int>(r.first));
        RecordProperty("Stddev", static_cast<int>(r.second));
    }
}
