// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "kamping/assertion_levels.hpp"
#undef KASSERT_ASSERTION_LEVEL
#define KASSERT_ASSERTION_LEVEL KAMPING_ASSERTION_LEVEL_HEAVY_COMMUNICATION

#include <set>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/environment.hpp"
using namespace ::kamping;

// This is not using google test because our test setup would call MPI_Init before running any tests
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    KASSERT(!mpi_env.initialized());
    KASSERT(!mpi_env.finalized());

#ifndef KAMPING_ENVIRONMENT_TEST_INIT_FINALIZE_NECESSARY
    MPI_Init(&argc, &argv);
#endif

    {
#if defined(KAMPING_ENVIRONMENT_TEST_NO_PARAM)
        Environment<InitMPIMode::InitFinalizeIfNecessary> environment;
#elif defined(KAMPING_ENVIRONMENT_TEST_WITH_PARAM)
        Environment<InitMPIMode::InitFinalizeIfNecessary> environment(argc, argv);
#else
        static_assert(false, "Define either KAMPING_ENVIRONMENT_TEST_NO_PARAM or KAMPING_ENVIRONMENT_TEST_WITH_PARAM");
#endif

        KASSERT(environment.initialized());
        KASSERT(!environment.finalized());

#ifndef KAMPING_ENVIRONMENT_TEST_INIT_FINALIZE_NECESSARY
        MPI_Finalize();
#endif
    }

    KASSERT(mpi_env.finalized());
    return 0;
}
