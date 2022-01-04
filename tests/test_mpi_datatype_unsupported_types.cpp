// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

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
    auto result = mpi_datatype<void>();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
