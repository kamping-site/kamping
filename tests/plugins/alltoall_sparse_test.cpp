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

#include "../test_assertions.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/alltoall_sparse.hpp"
#include "kamping/span.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;

TEST(ParameterFactoriesTest, sparse_send_buf_basics_with_non_container_object) {
    using namespace plugin::sparse_alltoall;
    struct TestStruct {
        int  a;
        int  b;
        int  c;
        bool operator==(TestStruct const& other) const {
            return std::tie(a, b, c) == std::tie(other.a, other.b, other.c);
        }
    };
    TestStruct const st{1, 2, 3};
    {
        // referencing sparse sparse send buf
        auto sparse_send_buffer = sparse_send_buf(st);
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_FALSE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), st);
    }
    {
        // referencing sparse sparse send buf initialized from non-const object
        TestStruct st_copy            = st;
        auto       sparse_send_buffer = sparse_send_buf(st_copy);
        using DataBufferType          = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_FALSE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), st);
    }
    {
        // owning sparse sparse send buf
        TestStruct st_copy            = st;
        auto       sparse_send_buffer = sparse_send_buf(std::move(st_copy));
        using DataBufferType          = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_TRUE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), st);
    }
    {
        // owning sparse sparse send buf via temporary
        auto sparse_send_buffer = sparse_send_buf(TestStruct{1, 2, 3});
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_TRUE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), st);
    }
}

TEST(ParameterFactoriesTest, sparse_send_buf_basics_with_unordered_map) {
    using namespace plugin::sparse_alltoall;
    std::unordered_map<int, std::vector<double>> const input{{1, {1.0, 2.0}}};
    {
        // referencing sparse sparse send buf
        auto sparse_send_buffer = sparse_send_buf(input);
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_FALSE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), input);
    }
    {
        // referencing sparse sparse send buf initialized from non-const object
        auto input_copy         = input;
        auto sparse_send_buffer = sparse_send_buf(input_copy);
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_FALSE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), input);
    }
    {
        // owning sparse sparse send buf
        auto input_copy         = input;
        auto sparse_send_buffer = sparse_send_buf(std::move(input_copy));
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_TRUE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), input);
    }
    {
        // owning sparse send buf via temporary
        auto sparse_send_buffer = sparse_send_buf(std::unordered_map<int, std::vector<double>>{{1, {1.0, 2.0}}});
        using DataBufferType    = decltype(sparse_send_buffer);
        EXPECT_EQ(DataBufferType::parameter_type, ParameterType::sparse_send_buf);
        EXPECT_TRUE(DataBufferType::is_owning);
        EXPECT_FALSE(DataBufferType::is_out_buffer);
        EXPECT_FALSE(DataBufferType::is_modifiable);
        EXPECT_EQ(sparse_send_buffer.underlying(), input);
    }
}

TEST(ParameterFactoriesTest, on_message_basics_lambda) {
    using namespace plugin::sparse_alltoall;
    int  state     = 0;
    auto add_value = [state](int val) mutable {
        state += val;
        return state;
    };
    {
        // referencing on message obj from non-const lvalue
        auto add_value_copy = add_value;
        auto on_message     = plugin::sparse_alltoall::on_message(add_value_copy);
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(42), 42);
        EXPECT_EQ(on_message.underlying()(1), 43);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_FALSE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj
        auto add_value_copy = add_value;
        auto on_message     = plugin::sparse_alltoall::on_message(std::move(add_value_copy));
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(42), 42);
        EXPECT_EQ(on_message.underlying()(1), 43);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj via temporary
        auto on_message     = plugin::sparse_alltoall::on_message([state](int val) mutable {
            state += val;
            return state;
        });
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(42), 42);
        EXPECT_EQ(on_message.underlying()(1), 43);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, on_message_basics_mutable_lambda) {
    using namespace plugin::sparse_alltoall;
    auto const cb = [](auto const&) {
        return 42;
    };
    {
        // referencing on message obj
        auto on_message     = plugin::sparse_alltoall::on_message(cb);
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(std::ignore), 42);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_FALSE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_FALSE(OnMessageType::is_modifiable);
    }
    {
        // referencing on message obj from non-const lvalue
        auto cb_copy        = cb;
        auto on_message     = plugin::sparse_alltoall::on_message(cb_copy);
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(std::ignore), 42);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_FALSE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj
        auto cb_copy        = cb;
        auto on_message     = plugin::sparse_alltoall::on_message(std::move(cb_copy));
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(std::ignore), 42);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj via temporary
        auto on_message     = plugin::sparse_alltoall::on_message([](auto const&) { return 42; });
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(std::ignore), 42);
        EXPECT_EQ(OnMessageType::parameter_type, ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, on_message_basics_callable_struct) {
    struct Callable {
        auto operator()() {
            return "nonconst-operator";
        }
        auto operator()() const {
            return "const-operator";
        }
        int state;
    };
    int const      state = 43;
    Callable const cb{state};
    {
        // referencing on message obj
        auto on_message     = plugin::sparse_alltoall::on_message(cb);
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(), "const-operator");
        EXPECT_EQ(on_message.underlying().state, state);
        EXPECT_EQ(OnMessageType::parameter_type, plugin::sparse_alltoall::ParameterType::on_message);
        EXPECT_FALSE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_FALSE(OnMessageType::is_modifiable);
    }
    {
        // referencing on message obj from non-const lvalue
        auto cb_copy        = cb;
        auto on_message     = plugin::sparse_alltoall::on_message(cb_copy);
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(), "nonconst-operator");
        EXPECT_EQ(on_message.underlying().state, state);
        EXPECT_EQ(OnMessageType::parameter_type, plugin::sparse_alltoall::ParameterType::on_message);
        EXPECT_FALSE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj
        auto cb_copy        = cb;
        auto on_message     = plugin::sparse_alltoall::on_message(std::move(cb_copy));
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(), "nonconst-operator");
        EXPECT_EQ(on_message.underlying().state, state);
        EXPECT_EQ(OnMessageType::parameter_type, plugin::sparse_alltoall::ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
    {
        // owning on message obj via temporary
        auto on_message     = plugin::sparse_alltoall::on_message(Callable{state});
        using OnMessageType = decltype(on_message);
        EXPECT_EQ(on_message.underlying()(), "nonconst-operator");
        EXPECT_EQ(on_message.underlying().state, state);
        EXPECT_EQ(OnMessageType::parameter_type, plugin::sparse_alltoall::ParameterType::on_message);
        EXPECT_TRUE(OnMessageType::is_owning);
        EXPECT_FALSE(OnMessageType::is_out_buffer);
        EXPECT_TRUE(OnMessageType::is_modifiable);
    }
}

TEST(SparseAlltoallTest, alltoallv_sparse_single_element) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    using namespace sparse_alltoall;
    Communicator<std::vector, plugin::SparseAlltoall> comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::vector<std::pair<int, msg_type>> input(comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace_back(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}
TEST(SparseAlltoallTest, alltoallv_sparse_single_element_with_map_as_sparse_send_buf) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto on_msg = [&](ProbedMessage<size_t, Communicator<std::vector, SparseAlltoall>> const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(SparseAlltoallTest, alltoallv_sparse_single_element_unordered_map_as_sparse_send_buf) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(
    SparseAlltoallTest, alltoallv_sparse_single_element_not_encapsulated_in_a_container_and_unordered_map_as_send_buf
) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type = size_t;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto on_msg = [&](ProbedMessage<size_t, Communicator<std::vector, SparseAlltoall>> const& probed_msg) {
        const int             source   = probed_msg.source_signed();
        std::vector<msg_type> recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(SparseAlltoallTest, alltoallv_sparse_one_to_all) {
    // Sends a message from rank 0 to all other ranks
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type        = std::vector<size_t>;
    size_t const msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), msg_size);
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(SparseAlltoallTest, alltoallv_sparse_one_to_all_recv_type_out) {
    // Sends a message from rank 0 to all other ranks
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type        = std::vector<size_t>;
    size_t const msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        const int source                 = probed_msg.source_signed();
        const auto [recv_msg, recv_type] = probed_msg.recv(recv_type_out());

        EXPECT_EQ(recv_msg.size(), msg_size);
        EXPECT_THAT(possible_mpi_datatypes<size_t>(), Contains(recv_type));
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(SparseAlltoallTest, alltoallv_sparse_one_to_all_recv_type_out_other_order) {
    // Sends a message from rank 0 to all other ranks
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type        = std::vector<size_t>;
    size_t const msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        const int source                 = probed_msg.source_signed();
        const auto [recv_type, recv_msg] = probed_msg.recv(recv_type_out(), kamping::recv_buf(alloc_new<msg_type>));

        EXPECT_EQ(recv_msg.size(), msg_size);
        EXPECT_THAT(possible_mpi_datatypes<size_t>(), Contains(recv_type));
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(SparseAlltoallTest, alltoallv_sparse_one_to_all_owning_send_buf_and_non_owning_recv_buf) {
    // Sends a message from rank 0 to all other ranks
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    using msg_type        = std::vector<size_t>;
    size_t const msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type(msg_size, i));
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](auto const& probed_msg) {
        msg_type recv_msg;
        probed_msg.recv(kamping::recv_buf<resize_to_fit>(recv_msg));
        EXPECT_EQ(recv_msg.size(), msg_size);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(std::move(input)), on_message(on_msg));

    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(SparseAlltoallTest, sparse_exchange) {
    // Send a message to left and right partner
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    if (comm.size() < 2) {
        return;
    }

    using msg_type = std::vector<size_t>;

    int const left_partner  = (comm.size_signed() + comm.rank_signed() - 1) % comm.size_signed();
    int const right_partner = (comm.rank_signed() + 1) % comm.size_signed();
    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    input.emplace(left_partner, msg_type(42, comm.rank()));
    input.emplace(right_partner, msg_type(42, comm.rank()));

    // Prepare cb
    std::unordered_map<int, msg_type> recv_buf;
    std::vector<int>                  sources;
    auto                              on_msg = [&](auto const& probed_msg) {
        auto recv_msg                        = probed_msg.recv();
        recv_buf[probed_msg.source_signed()] = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_EQ(recv_buf[left_partner], msg_type(42, asserting_cast<size_t>(left_partner)));
    EXPECT_EQ(recv_buf[right_partner], msg_type(42, asserting_cast<size_t>(right_partner)));
}

TEST(SparseAlltoallTest, alltoallv_sparse_sparse_exchange_custom_dynamic_send_datatype) {
    // Send a message to left and right partner
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    if (comm.size() < 2) {
        return;
    }
    struct Int_Padding_Int {
        int value_1;
        int padding;
        int value_2;
    };

    using msg_type               = Int_Padding_Int;
    MPI_Datatype int_padding_int = MPI_INT_padding_MPI_INT();
    MPI_Type_commit(&int_padding_int);

    int const left_partner  = (comm.size_signed() + comm.rank_signed() - 1) % comm.size_signed();
    int const right_partner = (comm.rank_signed() + 1) % comm.size_signed();
    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    input.emplace(left_partner, Int_Padding_Int{comm.rank_signed(), -1, 42});
    input.emplace(right_partner, Int_Padding_Int{comm.rank_signed(), -1, 42});

    // Prepare cb
    std::unordered_map<int, std::pair<int, int>> recv_messages;
    std::vector<int>                             sources;
    auto                                         on_msg = [&](auto const& probed_msg) {
        auto recv_msg = probed_msg.template recv<int>();
        EXPECT_EQ(probed_msg.recv_count(MPI_INT), 2);
        EXPECT_EQ(recv_msg.size(), 2);
        recv_messages[probed_msg.source_signed()] = std::make_pair(recv_msg.front(), recv_msg.back());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), send_type(int_padding_int), on_message(on_msg));

    EXPECT_EQ(recv_messages.size(), 2);
    EXPECT_EQ(recv_messages[left_partner], std::make_pair(left_partner, 42));
    EXPECT_EQ(recv_messages[right_partner], std::make_pair(right_partner, 42));

    MPI_Type_free(&int_padding_int);
}

TEST(SparseAlltoallTest, alltoallv_sparse_sparse_exchange_custom_dynamic_recv_datatype) {
    // Send a message to left and right partner
    using namespace plugin::sparse_alltoall;
    Communicator<std::vector, SparseAlltoall> comm;

    if (comm.size() < 2) {
        return;
    }

    int const msg_count = 42;
    using msg_type      = std::vector<int>;

    int const left_partner  = (comm.size_signed() + comm.rank_signed() - 1) % comm.size_signed();
    int const right_partner = (comm.rank_signed() + 1) % comm.size_signed();
    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    input.emplace(left_partner, msg_type(msg_count, comm.rank_signed()));
    input.emplace(right_partner, msg_type(msg_count, comm.rank_signed()));

    MPI_Datatype two_ints;
    MPI_Type_contiguous(2, MPI_INT, &two_ints);
    MPI_Type_commit(&two_ints);
    // Prepare cb
    std::unordered_map<int, msg_type> recv_buf;
    std::vector<int>                  sources;
    auto                              on_msg = [&](auto const& probed_msg) {
        msg_type recv_msg(msg_count);
        probed_msg.recv(kamping::recv_buf(recv_msg), recv_type(two_ints));
        EXPECT_EQ(probed_msg.recv_count(two_ints), msg_count / 2);
        EXPECT_EQ(probed_msg.recv_count(), msg_count);
        recv_buf[probed_msg.source_signed()] = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_EQ(recv_buf[left_partner], msg_type(msg_count, left_partner));
    EXPECT_EQ(recv_buf[right_partner], msg_type(msg_count, right_partner));

    MPI_Type_free(&two_ints);
}
