// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameters.hpp"
#include "legacy_parameter_objects.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    // none of the extract function should work if the underlying buffer does not provide a member extract().
    kamping::MPIResult mpi_result{
        StatusParam<StatusParamType::ignore>{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};

    constexpr BufferType                                                                  btype = BufferType::in_buffer;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_recv_buf;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_recv_buf;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_buf;
    auto result_recv_buf = make_mpi_result(
        std::move(recv_counts_recv_buf),
        std::move(recv_displs_recv_buf),
        std::move(send_displs_recv_buf)
    );

    LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_counts;
    auto result_recv_counts = make_mpi_result(
        std::move(recv_buf_recv_counts),
        std::move(recv_displs_recv_counts),
        std::move(send_displs_recv_counts)
    );

    LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_recv_displs;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_recv_displs;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_displs;
    auto result_recv_displs = make_mpi_result(
        std::move(recv_buf_recv_displs),
        std::move(recv_counts_recv_displs),
        std::move(send_displs_recv_displs)
    );

    LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_send_displs;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_send_displs;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_send_displs;
    auto result_send_displs = make_mpi_result(
        std::move(recv_buf_send_displs),
        std::move(recv_counts_send_displs),
        std::move(recv_displs_send_displs)
    );

#if defined(RECV_BUFFER_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_buffer();
#elif defined(RECV_COUNTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_counts();
#elif defined(RECV_DISPLACEMENTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_recv_displs();
#elif defined(SEND_DISPLACEMENTS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.extract_send_displs();
#elif defined(STATUS_NOT_EXTRACTABLE)
    std::ignore = mpi_result.status();
#elif defined(MAKE_MPI_RESULT_RECV_BUF_NOT_EXTRACTABLE)
    std::ignore = result_recv_buf.extract_recv_buffer();
#elif defined(MAKE_MPI_RESULT_RECV_COUNTS_NOT_EXTRACTABLE)
    std::ignore = result_recv_counts.extract_recv_counts();
#elif defined(MAKE_MPI_RESULT_RECV_DISPLS_NOT_EXTRACTABLE)
    std::ignore = result_recv_displs.extract_recv_displs();
#elif defined(MAKE_MPI_RESULT_SEND_COUNTS_NOT_EXTRACTABLE)
    std::ignore = result_send_displs.extract_send_displs();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
