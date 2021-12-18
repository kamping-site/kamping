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

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "kamping/buffer_factories.hpp"
#include "kamping/template_helpers.hpp"

/*
void printResult(int rank, std::vector<int>& recvData, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (auto elem: recvData) {
        ss << elem << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

void printResult(int rank, std::unique_ptr<int[]>& recvData, size_t size, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (size_t i = 0; i < size; ++i) {
        ss << recvData.get()[i] << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

int main() {
    MPI_Init(nullptr, nullptr);
    MPIWrapper::MPIContext ctx{MPI_COMM_WORLD};

    std::vector<int> sendData(static_cast<std::size_t>(ctx.rank() + 1), ctx.rank());
    // Gather sendData on PE 0 and allocate all other buffers inside the library
    auto gatherResults = ctx.gatherv(in(sendData));
    auto recvData      = gatherResults.extractRecvBuff();
    auto recvCounts    = gatherResults.extractRecvCounts();
    auto recvDispls    = gatherResults.extractRecvDispls();

    printResult(ctx.rank(), recvData, "data");
    printResult(ctx.rank(), recvCounts, "counts");
    printResult(ctx.rank(), recvDispls, "displs");

    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank() == 0) {
        std::cout << "----------------------------------------------" << std::endl;
    }
    // Sleep so all thread have time to flush their buffers
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> recvData2;
    // Gather sendData on PE 1, use existing vector to receive into and return the counts as a new unique_ptr
    auto gatherResults2 = ctx.gatherv(in(sendData), out(recvData2), root(1), recv_counts(new_pointer<int>()));

    // This fails because we have given the out vector as input
    // auto recvData3 = gatherResults2.extractRecvBuff();

    // unique_ptr because we requested it in the gatherv call
    std::unique_ptr<int[]> recvCounts2 = gatherResults2.extractRecvCounts();

    // default is vector
    std::vector<int> recvDispls2 = gatherResults2.extractRecvDispls();

    printResult(ctx.rank(), recvData2, "data");
    printResult(ctx.rank(), recvCounts2, static_cast<std::size_t>(ctx.rank() == 1 ? ctx.size() : 0), "counts");
    printResult(ctx.rank(), recvDispls2, "displs");

    MPI_Finalize();
}
*/

int main() {
    MPI_Init(nullptr, nullptr);
    {
        const std::vector<int> vec{1, 2, 3};
        auto                   send_buf = kamping::send_buf(vec);
        const auto [ptr, size]          = send_buf.get();
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << ptr[i] << std::endl;
        }
    }

    {
        const std::vector<int> vec{1, 2, 3};
        auto                   send_counts = kamping::send_counts(vec);
        const auto [ptr, size]             = send_counts.get();
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << ptr[i] << std::endl;
        }
    }

    {
        const std::vector<int> vec{1, 2, 3};
        auto                   recv_counts = kamping::recv_counts_given(vec);
        const auto [ptr, size]             = recv_counts.get();
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << ptr[i] << std::endl;
        }
    }

    {
        const std::vector<int> vec{1, 2, 3};
        auto                   send_displs = kamping::send_displs_given(vec);
        const auto [ptr, size]             = send_displs.get();
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << ptr[i] << std::endl;
        }
    }

    {
        const std::vector<int> vec{1, 2, 3};
        auto                   recv_displs = kamping::recv_displs_given(vec);
        const auto [ptr, size]             = recv_displs.get();
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << ptr[i] << std::endl;
        }
    }


    //{
    //    int  arr[3]            = {4, 5, 6};
    //    auto send_buf          = kamping::send_buf(arr, 3);
    //    const auto [ptr, size] = send_buf.get();
    //    for (std::size_t i = 0; i < size; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //}

    // const size_t n = 3;
    //{
    //     std::cout << "DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER" << std::endl;
    //     std::vector<int> vec{1, 2, 3};
    //     auto             send_displs = kamping::send_displs(vec);
    //     const auto       ptr         = send_displs.get_ptr(n);
    //     for (std::size_t i = 0; i < n; ++i) {
    //         std::cout << ptr[i] << std::endl;
    //     }
    //     std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    // }

    //{
    //    std::cout << "DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER" << std::endl;
    //    std::vector<int> vec{1, 2, 3};
    //    auto             send_displs = kamping::send_displs_input(vec);
    //    const auto       ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER" << std::endl;
    //    std::vector<int>       vec{1, 2, 3};
    //    std::unique_ptr<int[]> data(new int[n]());
    //    auto                   send_displs = kamping::send_displs_input(data);
    //    const auto             ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER" << std::endl;
    //    std::unique_ptr<int[]> data(new int[n]());
    //    auto                   send_displs = kamping::send_displs(data);
    //    const auto             ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER" << std::endl;
    //    auto       send_displs = kamping::send_displs(kamping::NewContainer<std::vector<int>>{});
    //    const auto ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER" << std::endl;
    //    auto       send_displs = kamping::send_displs(kamping::NewPtr<int>{});
    //    const auto ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_MOVED_CONTAINER_BASED_BUFFER" << std::endl;
    //    std::vector<int> vec{1, 2, 3};
    //    auto             send_displs = kamping::send_displs(std::move(vec));
    //    const auto       ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    //{
    //    std::cout << "DEFINE_MOVED_CONTAINER_BASED_BUFFER" << std::endl;
    //    std::vector<int> vec{1, 2, 3};
    //    auto             send_displs = kamping::send_displs_input(std::move(vec));
    //    const auto       ptr         = send_displs.get_ptr(n);
    //    for (std::size_t i = 0; i < n; ++i) {
    //        std::cout << ptr[i] << std::endl;
    //    }
    //    std::cout << "is consumable: " << decltype(send_displs)::is_consumable << std::endl;
    //}

    MPI_Finalize();
}
