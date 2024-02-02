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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "./helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/request_pool.hpp"

using namespace kamping;

TEST(RequestPoolTest, empty_pool) {
    kamping::RequestPool pool;
    pool.wait_all();
}

TEST(RequestPoolTest, wait_all) {
    kamping::RequestPool                            pool;
    std::vector<testing::DummyNonBlockingOperation> ops(5);
    std::vector<int>                                values;
    values.reserve(5);
    int i = 0;
    for (auto& op: ops) {
        values.emplace_back();
        op.start_op(kamping::request(pool.get_request()), kamping::tag(42 + i), recv_buf(values.back()));
        i++;
    }
    std::for_each(ops.begin(), ops.end(), [](auto& op) { op.finish_op(); });
    pool.wait_all();
    EXPECT_THAT(values, ::testing::ElementsAre(42, 43, 44, 45, 46));
}

TEST(RequestPoolTest, wait_all_statuses_out) {
    using namespace ::testing;
    kamping::RequestPool                   pool;
    std::vector<DummyNonBlockingOperation> ops(5);
    std::vector<int>                       values;
    values.reserve(5);
    int i = 0;
    for (auto& op: ops) {
        values.emplace_back();
        op.start_op(kamping::request(pool.get_request()), kamping::tag(42 + i), recv_buf(values.back()));
        i++;
    }
    std::for_each(ops.begin(), ops.end(), [](auto& op) { op.finish_op(); });
    std::vector<MPI_Status> statuses = pool.wait_all(statuses_out());
    EXPECT_THAT(values, ElementsAre(42, 43, 44, 45, 46));
    EXPECT_THAT(
        statuses,
        ElementsAre(
            Field(&MPI_Status::MPI_TAG, 42),
            Field(&MPI_Status::MPI_TAG, 43),
            Field(&MPI_Status::MPI_TAG, 44),
            Field(&MPI_Status::MPI_TAG, 45),
            Field(&MPI_Status::MPI_TAG, 46)
        )
    );
}

TEST(RequestPoolTest, wait_all_statuses_out_reference) {
    using namespace ::testing;
    kamping::RequestPool                   pool;
    std::vector<DummyNonBlockingOperation> ops(5);
    std::vector<int>                       values;
    values.reserve(5);
    int i = 0;
    for (auto& op: ops) {
        values.emplace_back();
        op.start_op(kamping::request(pool.get_request()), kamping::tag(42 + i), recv_buf(values.back()));
        i++;
    }
    std::for_each(ops.begin(), ops.end(), [](auto& op) { op.finish_op(); });
    std::vector<MPI_Status> statuses;
    pool.wait_all(statuses_out<resize_to_fit>(statuses));
    EXPECT_THAT(values, ElementsAre(42, 43, 44, 45, 46));
    EXPECT_THAT(
        statuses,
        ElementsAre(
            Field(&MPI_Status::MPI_TAG, 42),
            Field(&MPI_Status::MPI_TAG, 43),
            Field(&MPI_Status::MPI_TAG, 44),
            Field(&MPI_Status::MPI_TAG, 45),
            Field(&MPI_Status::MPI_TAG, 46)
        )
    );
}

TEST(RequestPoolTest, test_all) {
    using namespace ::testing;
    kamping::RequestPool      pool;
    DummyNonBlockingOperation op1;
    DummyNonBlockingOperation op2;
    int                       val1;
    int                       val2;
    op1.start_op(kamping::request(pool.get_request()), kamping::tag(42), recv_buf(val1));
    op2.start_op(kamping::request(pool.get_request()), kamping::tag(43), recv_buf(val2));
    EXPECT_THAT(pool.test_all(), A<bool>());
    EXPECT_FALSE(pool.test_all());
    op2.finish_op();
    EXPECT_FALSE(pool.test_all());
    op1.finish_op();
    EXPECT_TRUE(pool.test_all());
    EXPECT_EQ(val1, 42);
    EXPECT_EQ(val2, 43);
}

TEST(RequestPoolTest, test_all_statuses_out) {
    using namespace ::testing;
    kamping::RequestPool      pool;
    DummyNonBlockingOperation op1;
    DummyNonBlockingOperation op2;
    int                       val1;
    int                       val2;
    op1.start_op(kamping::request(pool.get_request()), kamping::tag(42), recv_buf(val1));
    op2.start_op(kamping::request(pool.get_request()), kamping::tag(43), recv_buf(val2));
    EXPECT_EQ(pool.test_all(statuses_out()), std::nullopt);
    op2.finish_op();
    EXPECT_EQ(pool.test_all(statuses_out()), std::nullopt);
    op1.finish_op();
    auto statuses = pool.test_all(statuses_out());
    EXPECT_THAT(statuses, Optional(ElementsAre(Field(&MPI_Status::MPI_TAG, 42), Field(&MPI_Status::MPI_TAG, 43))));
    EXPECT_EQ(val1, 42);
    EXPECT_EQ(val2, 43);
}

TEST(RequestPoolTest, test_all_statuses_out_reference) {
    using namespace ::testing;
    kamping::RequestPool      pool;
    DummyNonBlockingOperation op1;
    DummyNonBlockingOperation op2;
    int                       val1;
    int                       val2;
    op1.start_op(kamping::request(pool.get_request()), kamping::tag(42), recv_buf(val1));
    op2.start_op(kamping::request(pool.get_request()), kamping::tag(43), recv_buf(val2));
    std::vector<MPI_Status> statuses;
    EXPECT_THAT(pool.test_all(statuses_out<resize_to_fit>(statuses)), A<bool>());
    EXPECT_FALSE(pool.test_all(statuses_out<resize_to_fit>(statuses)));
    op2.finish_op();
    EXPECT_FALSE(pool.test_all(statuses_out<resize_to_fit>(statuses)));
    op1.finish_op();
    EXPECT_TRUE(pool.test_all(statuses_out<resize_to_fit>(statuses)));
    EXPECT_THAT(statuses, ElementsAre(Field(&MPI_Status::MPI_TAG, 42), Field(&MPI_Status::MPI_TAG, 43)));
    EXPECT_EQ(val1, 42);
    EXPECT_EQ(val2, 43);
}
