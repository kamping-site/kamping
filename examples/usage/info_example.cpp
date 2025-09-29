#include <ios>
#include "kamping/environment.hpp"
#include "kamping/info.hpp"

int main(int argc, char* argv[]) {
    kamping::Environment env;

    kamping::Info info;
    std::cout << std::boolalpha;
    std::cout << "contains? " << info.contains("foobar") << "\n";
    info.set("foobar", true);
    std::cout << "value: " << info.get("foobar").value_or("INVALID") << "\n";
    return 0;
}
