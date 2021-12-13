// This file includes the

#include "kamping/mpi_datatype.hpp"

using namespace ::kamping;

int main(int argc, char** argv) {
#if defined(POINTER)
    // Calling mpi_datatype with a pointer type should not compile.
    auto result = mpi_datatype<int*>();
#elif defined(FUNCTION)
    // Calling mpi_datatype with a function type should not compile.
    auto result = mpi_datatype<int(int)>();
#elif defined(UNION)
    // Calling mpi_datatype with a union type should not compile.
    union my_union {
        int a;
        int b;
    };
    auto result = mpi_datatype<my_union>();
#elif defined(VOID)
    // Calling mpi_datatype with a void type should not compile.
    auto result mpi_datatype<void>();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
