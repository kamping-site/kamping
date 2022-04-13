// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <mpi.h>

#include "kamping/environment.hpp"
#include "kamping/kassert.hpp"

using namespace ::kamping;

// This is not using google test because our test setup would call MPI_Init before running any tests
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    KASSERT(!mpi_env.initialized());
    KASSERT(!mpi_env.finalized());
    {
#if defined(KAMPING_ENVIRONMENT_TEST_NO_PARAM)
        Environment environment;
#elif defined(KAMPING_ENVIRONMENT_TEST_WITH_PARAM)
        Environment environment(argc, argv);
#else
        static_assert(false, "Define either KAMPING_ENVIRONMENT_TEST_NO_PARAM or KAMPING_ENVIRONMENT_TEST_WITH_PARAM");
#endif

        KASSERT(environment.initialized());
        KASSERT(!environment.finalized());
#if defined(KAMPING_ENVIRONMENT_TEST_EXPLICIT_FINALIZE)
        // Test that destructor works correctly even if finalize was called on a different object.
        mpi_env.finalize();
        KASSERT(environment.finalized());
#endif
        // If KAMPING_ENVIRONMENT_TEST_EXPLICIT_FINALIZE is not defined, MPI_Init() is called by `Environment`s
        // destructor after this closing bracket.
    }
    KASSERT(mpi_env.finalized());
    return 0;
}
