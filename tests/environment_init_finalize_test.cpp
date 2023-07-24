// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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

std::set<MPI_Datatype> freed_types;

int MPI_Type_free(MPI_Datatype* type) {
    freed_types.insert(*type);
    return PMPI_Type_free(type);
}

// This is not using google test because our test setup would call MPI_Init before running any tests
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    KASSERT(!mpi_env.initialized());
    KASSERT(!mpi_env.finalized());
    std::set<MPI_Datatype> types_to_be_freed;
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

        // Register MPI data types to be freed when finalizing
        MPI_Datatype type1, type2;
        MPI_Type_contiguous(1, MPI_CHAR, &type1);
        MPI_Type_commit(&type1);
        MPI_Type_contiguous(2, MPI_CHAR, &type2);
        MPI_Type_commit(&type2);
        MPI_Datatype type_null = MPI_DATATYPE_NULL;
        environment.register_mpi_type(type1);
        environment.register_mpi_type(type2);
        environment.register_mpi_type(type_null);
        types_to_be_freed.insert(type1);
        types_to_be_freed.insert(type2);

#if defined(KAMPING_ENVIRONMENT_TEST_EXPLICIT_FINALIZE)
        // Test that destructor works correctly even if finalize was called on a different object.
        mpi_env.finalize();
        KASSERT(environment.finalized());
#endif
        // If KAMPING_ENVIRONMENT_TEST_EXPLICIT_FINALIZE is not defined, MPI_Init() is called by `Environment`s
        // destructor after this closing bracket.
    }
    KASSERT(mpi_env.finalized());
    KASSERT(types_to_be_freed == freed_types);
    return 0;
}
