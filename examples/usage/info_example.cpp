#include <ios>
// #include "kamping/environment.hpp"
#include "kamping/info.hpp"

int main(int argc, char* argv[]) {
    // kamping::Environment env;

    kamping::Info info;
    std::cout << std::boolalpha;
    std::cout << "contains? " << info.contains("foobar") << "\n";
    info.set("foobar", true);
    info.set("foobar_num", 42);
    info.set("foobar_string", "hello");
    std::cout << "value: " << info.get("foobar").value_or("INVALID") << "\n";
    for (auto [key, value]: info) {
      std::cout << "<" << key << "," << value << ">" << "\n";
    }
    // std::find_if(key, value)
    return 0;
}
