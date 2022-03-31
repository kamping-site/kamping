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

#include <numeric>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

namespace testing {
// Mock object with extract method
struct StructWithExtract {
    void extract() {}
};

// Mock object without extract method
struct StructWithoutExtract {};

// Test that receive buffers can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_buffer_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_buffer = recv_buf(kamping::NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_buffer)::value_type>, "Use integral Types in this test.");

    recv_buffer.resize(10);
    int* ptr = recv_buffer.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        std::move(recv_buffer), BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_buffer();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that receive counts can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_counts_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_counts = recv_counts_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_counts)::value_type>, "Use integral Types in this test.");

    recv_counts.resize(10);
    int* ptr = recv_counts.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{}, std::move(recv_counts), BufferCategoryNotUsed{}, BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_counts();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that the receive count can be moved into and extracted from a MPIResult object.
void test_recv_count_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;

    RecvCount<int const> recv_count_wrapper = recv_count(42);
    MPIResult            mpi_result{
        BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, std::move(recv_count_wrapper), BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
    int recv_count_value = mpi_result.extract_recv_count();
    EXPECT_EQ(recv_count_value, 42);
}

// Test that receive displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_displs = recv_displs_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_displs)::value_type>, "Use integral Types in this test.");

    recv_displs.resize(10);
    int* ptr = recv_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, std::move(recv_displs),
        BufferCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that send displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_displs = send_displs_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(send_displs)::value_type>, "Use integral Types in this test.");

    send_displs.resize(10);
    int* ptr = send_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{},
        std::move(send_displs)};
    UnderlyingContainer underlying_container = mpi_result.extract_send_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}
} // namespace testing


TEST(MpiResultTest, has_extract_v_basics) {
    static_assert(
        has_extract_v<testing::StructWithExtract>,
        "StructWithExtract contains extract() member function -> needs to be detected.");
    static_assert(
        !has_extract_v<testing::StructWithoutExtract>,
        "StructWithoutExtract does not contain extract() member function.");
}

TEST(MpiResultTest, extract_recv_buffer_basics) {
    testing::test_recv_buffer_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_buffer_basics_own_container) {
    testing::test_recv_buffer_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics) {
    testing::test_recv_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics_own_container) {
    testing::test_recv_counts_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_count_basics) {
    testing::test_recv_count_in_MPIResult();
}

TEST(MpiResultTest, extract_recv_displs_basics) {
    testing::test_recv_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_displs_basics_own_container) {
    testing::test_recv_displs_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics) {
    testing::test_send_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics_own_container) {
    testing::test_send_displs_in_MPIResult<testing::OwnContainer<int>>();
}

/// @brief Mixin for addition operation
template <class Communicator>
class Adder : public CRTPHelper<Communicator, Adder> {
public:
    int root_plus(int x) const {
        return this->underlying().root() + x;
    }
};

/// @brief Mixin for multiplication operation
template <class Communicator>
class Multiplier : public CRTPHelper<Communicator, Multiplier> {
public:
    int root_times(int x) const {
        return this->underlying().root() * x;
    }
};


/// @brief Mixin for Changing the root
template <class Communicator>
class RootChanger : public CRTPHelper<Communicator, RootChanger> {
public:
    void change_root(int new_root) {
        return this->underlying().root(new_root);
    }
};

/// @brief dummy communicator for testing CRTPHelper
class DummyCommunicator : public Adder<DummyCommunicator>,
                          public Multiplier<DummyCommunicator>,
                          public RootChanger<DummyCommunicator> {
public:
    DummyCommunicator(int root) : _root(root) {}
    int root() const {
        return _root;
    }

    void root(int new_root) {
        _root = new_root;
    }

private:
    int _root;
};

TEST(CRTPHelperTest, crtp_works) {
    DummyCommunicator comm{42};
    // Check that constructor and getting the root actually works
    EXPECT_EQ(comm.root(), 42);
    // Check that const-access mixins work
    EXPECT_EQ(comm.root_plus(3), 42 + 3);
    EXPECT_EQ(comm.root_times(3), 42 * 3);

    // Check that changing the root works (this is directly from DummyCommunicator)
    comm.root(0);
    EXPECT_EQ(comm.root(), 0);

    // Check that non-const-access mixins work (this is implemented in RootChanger)
    comm.change_root(69);
    EXPECT_EQ(comm.root(), 69);
    EXPECT_EQ(comm.root_plus(2), 69 + 2);
    EXPECT_EQ(comm.root_times(2), 69 * 2);
}
