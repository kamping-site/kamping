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

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/serialization.hpp"

using namespace kamping;

using dict_type = std::unordered_map<std::string, std::string>;

TEST(SerializationTest, basic) {
    kamping::Communicator comm;
    dict_type             data{{"key1", "value1"}, {"key2", "value2"}};
    if (comm.is_root()) {
        for (size_t dst = 0; dst < comm.size(); dst++) {
            if (comm.is_root(dst)) {
                continue;
            }
            comm.send(kamping::send_buf(as_serialized(data)), destination(dst));
        }
    } else {
        auto recv_data = comm.recv(recv_buf(as_deserializable<dict_type>())).deserialize();
        EXPECT_EQ(recv_data, data);
    }
}

TEST(SerializationTest, no_explicit_deserialization_type) {
    kamping::Communicator comm;
    dict_type             data{{"key1", "value1"}, {"key2", "value2"}};
    if (comm.is_root()) {
        for (size_t dst = 0; dst < comm.size(); dst++) {
            if (comm.is_root(dst)) {
                continue;
            }
            comm.send(kamping::send_buf(as_serialized(data)), destination(dst));
        }
    } else {
        auto recv_data = comm.recv(recv_buf(as_deserializable())).deserialize<dict_type>();
        EXPECT_EQ(recv_data, data);
    }
}

TEST(SerializationTest, no_explicit_non_default_archive) {
    kamping::Communicator comm;
    dict_type             data{{"key1", "value1"}, {"key2", "value2"}};
    if (comm.is_root()) {
        for (size_t dst = 0; dst < comm.size(); dst++) {
            if (comm.is_root(dst)) {
                continue;
            }
            comm.send(kamping::send_buf(as_serialized<cereal::JSONOutputArchive>(data)), destination(dst));
        }
    } else {
        auto recv_data = comm.recv(recv_buf(as_deserializable<dict_type, cereal::JSONInputArchive>())).deserialize();
        EXPECT_EQ(recv_data, data);
    }
}

struct Foo {
    double           x;
    std::vector<int> v;
    template <class Archive>
    void serialize(Archive& ar) {
        ar(x, v);
    }
    bool operator==(Foo const& other) const {
        return this->x == other.x && this->v == other.v;
    }
};

TEST(SerializationTest, custom_serialization_functions) {
    kamping::Communicator comm;
    Foo                   data = {3.14, {1, 2, 3}};
    if (comm.is_root()) {
        for (size_t dst = 0; dst < comm.size(); dst++) {
            if (comm.is_root(dst)) {
                continue;
            }
            comm.send(kamping::send_buf(as_serialized(data)), destination(dst));
        }
    } else {
        auto recv_data = comm.recv(recv_buf(as_deserializable<Foo>())).deserialize();
        EXPECT_EQ(recv_data, data);
    }
}
