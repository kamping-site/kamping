#include <unordered_map>

#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>

#include "cereal/archives/binary.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/unordered_map.hpp"

template <typename T>
auto as_serialized(T const& data) {
    std::stringstream buffer;
    {
        cereal::BinaryOutputArchive archive(buffer);
        archive(data);
    }
    return std::move(buffer).str();
}

template <typename T>
struct DeserializableBuffer {
    std::vector<char> _data;
    T                 deserialize() {
                        std::istringstream buffer(std::string(_data.begin(), _data.end()));
                        T                  result;
                        {
                            cereal::BinaryInputArchive archive(buffer);
                            archive(result);
        }
                        return result;
    }
    using value_type = char;
    char* data() noexcept {
        return _data.data();
    }

    void resize(size_t size) {
        _data.resize(size);
    }

    size_t size() const {
        return _data.size();
    }
};

int main() {
    kamping::Environment                         env;
    std::unordered_map<std::string, std::string> data = {{"key1", "value1"}, {"key2", "value2"}};
    auto const&                                  comm = kamping::comm_world();
    if (comm.rank() == 0) {
        comm.send(kamping::send_buf(as_serialized(data)), kamping::destination(0));
        DeserializableBuffer<std::unordered_map<std::string, std::string>> b;
        auto result = comm.recv(kamping::recv_buf<kamping::resize_to_fit>(std::move(b))).deserialize();
        for (auto const& [key, value]: result) {
            std::cout << key << " -> " << value << std::endl;
        }
    }

    return 0;
}
