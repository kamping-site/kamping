// This file is part of KaMPI.ng.
//
// Copyright 2024 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <unordered_map>

#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/serialization.hpp>

#include "cereal/types/string.hpp"
#include "cereal/types/unordered_map.hpp"

int main() {
    kamping::Environment env;
    using dict_type  = std::unordered_map<std::string, std::string>;
    dict_type   data = {{"key1", "value1"}, {"key2", "value2"}};
    auto const& comm = kamping::comm_world();
    if (comm.rank() == 0) {
        comm.send(kamping::send_buf(kamping::as_serialized(data)), kamping::destination(0));
        auto result = comm.recv(kamping::recv_buf(kamping::as_deserializable<dict_type>()));
        for (auto const& [key, value]: result) {
            std::cout << key << " -> " << value << std::endl;
        }
    }

    return 0;
}
