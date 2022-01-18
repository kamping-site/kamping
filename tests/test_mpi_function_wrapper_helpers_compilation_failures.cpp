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

#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/parameter_factories.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    // none of the extract function should work if the underlying buffer does not provide a member extract().
    kamping::MPIResult mpi_result{
        BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{}};
#if defined(RECV_BUFFER_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_buffer();
#elif defined(RECV_COUNTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_counts();
#elif defined(RECV_DISPLACEMENTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_displs();
#elif defined(SEND_DISPLACEMENTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_send_displs();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
