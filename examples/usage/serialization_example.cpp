#include <unordered_map>

#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/serialization.hpp>

#include "cereal/types/string.hpp"
#include "cereal/types/unordered_map.hpp"

int main() {
    kamping::Environment                         env;
    std::unordered_map<std::string, std::string> data = {{"key1", "value1"}, {"key2", "value2"}};
    auto const&                                  comm = kamping::comm_world();
    using kv_type                                     = decltype(data);
    if (comm.rank() == 0) {
        comm.send(kamping::send_buf(kamping::as_serialized(data)), kamping::destination(0));
        auto result = comm.recv(kamping::recv_buf(kamping::as_deserializable<kv_type>())).deserialize();
        for (auto const& [key, value]: result) {
            std::cout << key << " -> " << value << std::endl;
        }
    }

    return 0;
}
