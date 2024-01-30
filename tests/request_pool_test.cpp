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

#include "kamping/communicator.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/request_pool.hpp"
#include "kamping/status_vector.hpp"

using namespace kamping;

struct DummyNonBlockingOperations {
    template <typename... Args>
    auto start_op(Args... args) {
        using namespace kamping::internal;
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(),
            KAMPING_OPTIONAL_PARAMETERS(tag, request, recv_buf)
        );
        using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<std::vector<int>>));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            );
        using recv_buf_type       = typename std::remove_reference_t<decltype(recv_buf)>;
        using recv_buf_value_type = typename recv_buf_type::value_type;
        static_assert(std::is_same_v<recv_buf_value_type, int>);
        auto compute_required_recv_buf_size = [&] {
            return size_t{1};
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        using default_request_param = decltype(kamping::request());
        auto&& request_param =
            internal::select_parameter_type_or_default<internal::ParameterType::request, default_request_param>(
                std::tuple{},
                args...
            );

        using default_tag_buf_type = decltype(kamping::tag(0));

        auto&& tag_param =
            internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>(
                std::tuple(0),
                args...
            );
        int tag     = tag_param.tag();
        this->state = new int(tag);
        this->data  = recv_buf.get().data();
        MPI_Grequest_start(
            [](void* extra_state [[maybe_unused]], MPI_Status* status [[maybe_unused]]) {
                MPI_Status_set_elements(status, MPI_INT, 1);
                MPI_Status_set_cancelled(status, 0);
                MPI_Comm_rank(MPI_COMM_WORLD, &status->MPI_SOURCE);
                status->MPI_TAG = *static_cast<int*>(extra_state);
                return MPI_SUCCESS;
            },
            [](void* extra_state [[maybe_unused]]) {
                delete static_cast<int*>(extra_state);
                return MPI_SUCCESS;
            },
            [](void* extra_state [[maybe_unused]], int complete [[maybe_unused]]) { return MPI_SUCCESS; },
            this->state,
            &request_param.underlying().mpi_request()
        );
        this->req = request_param.underlying().mpi_request();
        return make_nonblocking_result(std::move(recv_buf), std::move(request_param));
    }
    void finish_op() {
        *this->data = *state;
        MPI_Grequest_complete(this->req);
    }
    int*        state;
    int*        data;
    MPI_Request req;
};

TEST(RequestPoolTest, empty_pool) {
    kamping::RequestPool pool;
    pool.wait_all();
}

TEST(RequestPoolTest, wait_all) {
    kamping::RequestPool                    pool;
    std::vector<DummyNonBlockingOperations> ops(5);
    std::vector<int>                        values;
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
    kamping::RequestPool                    pool;
    std::vector<DummyNonBlockingOperations> ops(5);
    std::vector<int>                        values;
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
    kamping::RequestPool                    pool;
    std::vector<DummyNonBlockingOperations> ops(5);
    std::vector<int>                        values;
    values.reserve(5);
    int i = 0;
    for (auto& op: ops) {
        values.emplace_back();
        op.start_op(kamping::request(pool.get_request()), kamping::tag(42 + i), recv_buf(values.back()));
        i++;
    }
    std::for_each(ops.begin(), ops.end(), [](auto& op) { op.finish_op(); });
    kamping::status_vector statuses;
    // std::vector<MPI_Status> statuses;
    pool.wait_all(statuses_out<resize_to_fit>(statuses));
    EXPECT_THAT(values, ElementsAre(42, 43, 44, 45, 46));
    EXPECT_THAT(
        statuses,
        ElementsAre(
            Property(&Status::tag, 42),
            Property(&Status::tag, 43),
            Property(&Status::tag, 44),
            Property(&Status::tag, 45),
            Property(&Status::tag, 46)
        )
    );
}
