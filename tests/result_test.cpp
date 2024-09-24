// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "gmock/gmock.h"
#include <numeric>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "legacy_parameter_objects.hpp"

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
    auto construct_recv_buffer = []() {
        auto recv_buffer = recv_buf(kamping::alloc_new<UnderlyingContainer>).construct_buffer_or_rebind();
        static_assert(
            std::is_integral_v<typename decltype(recv_buffer)::value_type>,
            "Use integral Types in this test."
        );

        recv_buffer.resize(10);
        int* ptr = recv_buffer.data();
        std::iota(ptr, ptr + 10, 0);
        return recv_buffer;
    };

    {
        MPIResult           mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer underlying_container = mpi_result.extract_recv_buffer();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult           mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer underlying_container = mpi_result.extract_recv_buf();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer& underlying_container = mpi_result.get_recv_buffer();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer& underlying_container = mpi_result.get_recv_buf();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer& underlying_container =
            mpi_result.template get<decltype(construct_recv_buffer())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_recv_buffer())};
        mpi_result.get_recv_buffer();
        UnderlyingContainer const& underlying_container = mpi_result.get_recv_buffer();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_recv_buffer())};
        mpi_result.get_recv_buffer();
        UnderlyingContainer const& underlying_container = mpi_result.get_recv_buf();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_recv_buffer())};
        UnderlyingContainer const& underlying_container =
            mpi_result.template get<decltype(construct_recv_buffer())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
}

// Test that receive counts can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_counts_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_recv_counts = []() {
        auto recv_counts = recv_counts_out(alloc_new<UnderlyingContainer>).construct_buffer_or_rebind();
        static_assert(
            std::is_integral_v<typename decltype(recv_counts)::value_type>,
            "Use integral Types in this test."
        );

        recv_counts.resize(10);
        int* ptr = recv_counts.data();
        std::iota(ptr, ptr + 10, 0);
        return recv_counts;
    };
    {
        MPIResult           mpi_result{std::make_tuple(construct_recv_counts())};
        UnderlyingContainer underlying_container = mpi_result.extract_recv_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_counts())};
        UnderlyingContainer& underlying_container = mpi_result.get_recv_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_counts())};
        UnderlyingContainer& underlying_container =
            mpi_result.template get<decltype(construct_recv_counts())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_recv_counts())};
        UnderlyingContainer const& underlying_container = mpi_result.get_recv_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_recv_counts())};
        UnderlyingContainer const& underlying_container =
            mpi_result.template get<decltype(construct_recv_counts())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
}

// Test that receive displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_recv_displs = []() {
        auto recv_displs = recv_displs_out(alloc_new<UnderlyingContainer>).construct_buffer_or_rebind();
        static_assert(
            std::is_integral_v<typename decltype(recv_displs)::value_type>,
            "Use integral Types in this test."
        );

        recv_displs.resize(10);
        int* ptr = recv_displs.data();
        std::iota(ptr, ptr + 10, 0);
        return recv_displs;
    };
    {
        MPIResult           mpi_result{std::make_tuple(construct_recv_displs())};
        UnderlyingContainer underlying_container = mpi_result.extract_recv_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_displs())};
        UnderlyingContainer& underlying_container = mpi_result.get_recv_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_recv_displs())};
        UnderlyingContainer& underlying_container =
            mpi_result.template get<decltype(construct_recv_displs())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_recv_displs())};
        UnderlyingContainer const& underlying_container = mpi_result.get_recv_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_recv_displs())};
        UnderlyingContainer const& underlying_container =
            mpi_result.template get<decltype(construct_recv_displs())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
}

// Test that send counts can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_counts_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_send_counts = []() {
        auto send_counts = send_counts_out(alloc_new<UnderlyingContainer>).construct_buffer_or_rebind();
        static_assert(
            std::is_integral_v<typename decltype(send_counts)::value_type>,
            "Use integral Types in this test."
        );

        send_counts.resize(10);
        int* ptr = send_counts.data();
        std::iota(ptr, ptr + 10, 0);
        return send_counts;
    };
    {
        MPIResult           mpi_result{std::make_tuple(construct_send_counts())};
        UnderlyingContainer underlying_container = mpi_result.extract_send_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_send_counts())};
        UnderlyingContainer& underlying_container = mpi_result.get_send_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_send_counts())};
        UnderlyingContainer& underlying_container =
            mpi_result.template get<decltype(construct_send_counts())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_send_counts())};
        UnderlyingContainer const& underlying_container = mpi_result.get_send_counts();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_send_counts())};
        UnderlyingContainer const& underlying_container =
            mpi_result.template get<decltype(construct_send_counts())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
}

// Test that send displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_send_displs = []() {
        auto send_displs = send_displs_out(alloc_new<UnderlyingContainer>).construct_buffer_or_rebind();
        static_assert(
            std::is_integral_v<typename decltype(send_displs)::value_type>,
            "Use integral Types in this test."
        );

        send_displs.resize(10);
        int* ptr = send_displs.data();
        std::iota(ptr, ptr + 10, 0);
        return send_displs;
    };
    {
        MPIResult           mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer underlying_container = mpi_result.extract_send_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer& underlying_container = mpi_result.get_send_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer& underlying_container = mpi_result.get_send_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult            mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer& underlying_container =
            mpi_result.template get<decltype(construct_send_displs())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer const& underlying_container = mpi_result.get_send_displs();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
    {
        MPIResult const            mpi_result{std::make_tuple(construct_send_displs())};
        UnderlyingContainer const& underlying_container =
            mpi_result.template get<decltype(construct_send_displs())::parameter_type>();
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_EQ(underlying_container[i], i);
        }
    }
}

KAMPING_MAKE_HAS_MEMBER(extract)
/// @brief has_extract_v is \c true iff type T has a member function \c extract().
///
/// @tparam T Type which is tested for the existence of a member function.
template <typename T>
inline constexpr bool has_extract_v = has_member_extract_v<T>;

} // namespace testing
  //

TEST(MpiResultTest, has_extract_v_basics) {
    static_assert(
        testing::has_extract_v<::testing::StructWithExtract>,
        "StructWithExtract contains extract() member function -> needs to be detected."
    );
    static_assert(
        !testing::has_extract_v<::testing::StructWithoutExtract>,
        "StructWithoutExtract does not contain extract() member function."
    );
}

TEST(MpiResultTest, extract_recv_buffer_basics) {
    ::testing::test_recv_buffer_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_buffer_basics_own_container) {
    ::testing::test_recv_buffer_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics) {
    ::testing::test_recv_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics_own_container) {
    ::testing::test_recv_counts_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_displs_basics) {
    ::testing::test_recv_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_displs_basics_own_container) {
    ::testing::test_recv_displs_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_counts_basics) {
    ::testing::test_send_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_counts_basics_own_container) {
    ::testing::test_send_counts_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics) {
    ::testing::test_send_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics_own_container) {
    ::testing::test_send_displs_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_count) {
    using namespace kamping;
    using namespace kamping::internal;
    auto const construct_send_count = []() {
        LibAllocatedSingleElementBuffer<int, ParameterType::send_count, BufferType::out_buffer> send_count_wrapper{};
        send_count_wrapper.underlying() = 42;
        return send_count_wrapper;
    };

    {
        MPIResult mpi_result{std::make_tuple(construct_send_count())};
        int       send_count = mpi_result.extract_send_count();
        EXPECT_EQ(send_count, 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_send_count())};
        int&      send_count = mpi_result.get_send_count();
        EXPECT_EQ(send_count, 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_send_count())};
        int&      send_count = mpi_result.template get<decltype(construct_send_count())::parameter_type>();
        EXPECT_EQ(send_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_send_count())};
        int const&      send_count = mpi_result.get_send_count();
        EXPECT_EQ(send_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_send_count())};
        int const&      send_count = mpi_result.template get<decltype(construct_send_count())::parameter_type>();
        EXPECT_EQ(send_count, 42);
    }
}

TEST(MpiResultTest, extract_recv_count) {
    using namespace kamping;
    using namespace kamping::internal;
    auto const construct_recv_count = []() {
        LibAllocatedSingleElementBuffer<int, ParameterType::recv_count, BufferType::out_buffer> recv_count_wrapper{};
        recv_count_wrapper.underlying() = 42;
        return recv_count_wrapper;
    };

    {
        MPIResult mpi_result{std::make_tuple(construct_recv_count())};
        int       recv_count = mpi_result.extract_recv_count();
        EXPECT_EQ(recv_count, 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_recv_count())};
        int&      recv_count = mpi_result.get_recv_count();
        EXPECT_EQ(recv_count, 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_recv_count())};
        int&      recv_count = mpi_result.template get<decltype(construct_recv_count())::parameter_type>();
        EXPECT_EQ(recv_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_recv_count())};
        int const&      recv_count = mpi_result.get_recv_count();
        EXPECT_EQ(recv_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_recv_count())};
        int const&      recv_count = mpi_result.template get<decltype(construct_recv_count())::parameter_type>();
        EXPECT_EQ(recv_count, 42);
    }
}

TEST(MpiResultTest, extract_send_recv_count) {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_send_recv_count = []() {
        auto send_recv_count         = kamping::send_recv_count_out().construct_buffer_or_rebind();
        send_recv_count.underlying() = 42;
        return send_recv_count;
    };
    {
        MPIResult mpi_result{std::make_tuple(construct_send_recv_count())};
        EXPECT_EQ(mpi_result.extract_send_recv_count(), 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_send_recv_count())};
        int&      send_recv_count = mpi_result.get_send_recv_count();
        EXPECT_EQ(send_recv_count, 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_send_recv_count())};
        int&      send_recv_count = mpi_result.template get<decltype(construct_send_recv_count())::parameter_type>();
        EXPECT_EQ(send_recv_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_send_recv_count())};
        int const&      send_recv_count = mpi_result.get_send_recv_count();
        EXPECT_EQ(send_recv_count, 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_send_recv_count())};
        int const& send_recv_count = mpi_result.template get<decltype(construct_send_recv_count())::parameter_type>();
        EXPECT_EQ(send_recv_count, 42);
    }
}

TEST(MpiResultTest, extract_send_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_send_type = []() {
        auto send_type         = kamping::send_type_out().construct_buffer_or_rebind();
        send_type.underlying() = MPI_DOUBLE;
        return send_type;
    };
    {
        MPIResult mpi_result{std::make_tuple(construct_send_type())};
        EXPECT_EQ(mpi_result.extract_send_type(), MPI_DOUBLE);
    }
    {
        MPIResult     mpi_result{std::make_tuple(construct_send_type())};
        MPI_Datatype& send_type = mpi_result.get_send_type();
        EXPECT_EQ(send_type, MPI_DOUBLE);
    }
    {
        MPIResult     mpi_result{std::make_tuple(construct_send_type())};
        MPI_Datatype& send_type = mpi_result.template get<decltype(construct_send_type())::parameter_type>();
        EXPECT_EQ(send_type, MPI_DOUBLE);
    }
    {
        MPIResult const     mpi_result{std::make_tuple(construct_send_type())};
        MPI_Datatype const& send_type = mpi_result.get_send_type();
        EXPECT_EQ(send_type, MPI_DOUBLE);
    }
    {
        MPIResult const     mpi_result{std::make_tuple(construct_send_type())};
        MPI_Datatype const& send_type = mpi_result.template get<decltype(construct_send_type())::parameter_type>();
        EXPECT_EQ(send_type, MPI_DOUBLE);
    }
}

TEST(MpiResultTest, extract_recv_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_recv_type = []() {
        auto recv_type         = kamping::recv_type_out().construct_buffer_or_rebind();
        recv_type.underlying() = MPI_CHAR;
        return recv_type;
    };
    {
        MPIResult mpi_result{std::tuple(construct_recv_type())};
        EXPECT_EQ(mpi_result.extract_recv_type(), MPI_CHAR);
    }
    {
        MPIResult     mpi_result{std::tuple(construct_recv_type())};
        MPI_Datatype& recv_type = mpi_result.get_recv_type();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
    {
        MPIResult     mpi_result{std::tuple(construct_recv_type())};
        MPI_Datatype& recv_type = mpi_result.template get<decltype(construct_recv_type())::parameter_type>();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }

    {
        MPIResult const     mpi_result{std::tuple(construct_recv_type())};
        MPI_Datatype const& recv_type = mpi_result.get_recv_type();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
    {
        MPIResult const     mpi_result{std::tuple(construct_recv_type())};
        MPI_Datatype const& recv_type = mpi_result.template get<decltype(construct_recv_type())::parameter_type>();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
}

TEST(MpiResultTest, extract_send_recv_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_send_recv_type = []() {
        auto send_recv_type         = kamping::send_recv_type_out().construct_buffer_or_rebind();
        send_recv_type.underlying() = MPI_CHAR;
        return send_recv_type;
    };
    {
        MPIResult mpi_result{std::tuple(construct_send_recv_type())};
        EXPECT_EQ(mpi_result.extract_send_recv_type(), MPI_CHAR);
    }
    {
        MPIResult     mpi_result{std::tuple(construct_send_recv_type())};
        MPI_Datatype& recv_type = mpi_result.get_send_recv_type();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
    {
        MPIResult     mpi_result{std::tuple(construct_send_recv_type())};
        MPI_Datatype& recv_type = mpi_result.template get<decltype(construct_send_recv_type())::parameter_type>();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
    {
        MPIResult const     mpi_result{std::tuple(construct_send_recv_type())};
        MPI_Datatype const& recv_type = mpi_result.get_send_recv_type();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
    {
        MPIResult const     mpi_result{std::tuple(construct_send_recv_type())};
        MPI_Datatype const& recv_type = mpi_result.template get<decltype(construct_send_recv_type())::parameter_type>();
        EXPECT_EQ(recv_type, MPI_CHAR);
    }
}

TEST(MpiResultTest, extract_status_basics) {
    using namespace kamping;
    using namespace kamping::internal;
    auto construct_status = []() {
        auto status = status_out().construct_buffer_or_rebind();

        status_param_to_native_ptr(status)->MPI_TAG = 42;
        return status;
    };
    {
        MPIResult mpi_result{std::make_tuple(construct_status())};
        auto      underlying_status = mpi_result.extract_status();
        EXPECT_EQ(underlying_status.tag(), 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_status())};
        auto&     underlying_status = mpi_result.get_status();
        EXPECT_EQ(underlying_status.tag(), 42);
    }
    {
        MPIResult mpi_result{std::make_tuple(construct_status())};
        auto&     underlying_status = mpi_result.template get<decltype(construct_status())::parameter_type>();
        EXPECT_EQ(underlying_status.tag(), 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_status())};
        auto const&     underlying_status = mpi_result.get_status();
        EXPECT_EQ(underlying_status.tag(), 42);
    }
    {
        MPIResult const mpi_result{std::make_tuple(construct_status())};
        auto const&     underlying_status = mpi_result.template get<decltype(construct_status())::parameter_type>();
        EXPECT_EQ(underlying_status.tag(), 42);
    }
}

KAMPING_MAKE_HAS_MEMBER(extract_status)
KAMPING_MAKE_HAS_MEMBER(extract_recv_buffer)
KAMPING_MAKE_HAS_MEMBER(extract_recv_counts)
KAMPING_MAKE_HAS_MEMBER(extract_recv_count)
KAMPING_MAKE_HAS_MEMBER(extract_recv_displs)
KAMPING_MAKE_HAS_MEMBER(extract_send_counts)
KAMPING_MAKE_HAS_MEMBER(extract_send_count)
KAMPING_MAKE_HAS_MEMBER(extract_send_displs)
KAMPING_MAKE_HAS_MEMBER(extract_send_recv_count)
KAMPING_MAKE_HAS_MEMBER(extract_send_type)
KAMPING_MAKE_HAS_MEMBER(extract_recv_type)
KAMPING_MAKE_HAS_MEMBER(extract_send_recv_type)

TEST(MpiResultTest, removed_extract_functions) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    constexpr BufferType btype = BufferType::out_buffer;
    {
        // All of these should be extractable (used to make sure that the above macros work correctly)
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::recv_count, btype>               recv_count_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_count, btype>               send_count_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_recv_count, btype>    send_recv_count_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_type, btype> send_type_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::recv_type, btype> recv_type_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_recv_type, btype>
                           send_recv_type_sanity_check;
        kamping::MPIResult mpi_result_sanity_check{std::make_tuple(
            std::move(status_sanity_check),
            std::move(recv_buf_sanity_check),
            std::move(recv_counts_sanity_check),
            std::move(recv_count_sanity_check),
            std::move(recv_displs_sanity_check),
            std::move(send_counts_sanity_check),
            std::move(send_count_sanity_check),
            std::move(send_displs_sanity_check),
            std::move(send_recv_count_sanity_check),
            std::move(send_type_sanity_check),
            std::move(recv_type_sanity_check),
            std::move(send_recv_type_sanity_check)
        )};
        EXPECT_TRUE(has_member_extract_status_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_recv_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_recv_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_FALSE(decltype(mpi_result_sanity_check)::is_empty);
    }

    {
        // none of the extract function should work if the underlying buffer does not provide a member extract().
        kamping::MPIResult mpi_result{std::make_tuple()};
        EXPECT_FALSE(has_member_extract_status_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_counts_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_displs_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_counts_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_displs_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_recv_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_type_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_type_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_recv_type_v<decltype(mpi_result)>);
        EXPECT_TRUE(decltype(mpi_result)::is_empty);
    }

    {
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype>,
            LibAllocatedContainerBasedBuffer<int, ParameterType::send_count, btype>,
            LibAllocatedContainerBasedBuffer<int, ParameterType::recv_count, btype>,
            LibAllocatedContainerBasedBuffer<int, ParameterType::send_recv_count, btype>,
            LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_type, btype>,
            LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::recv_type, btype>,
            LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_recv_type, btype>>;

        std::tuple_element_t<0, OutParameters>  recv_buf_status;
        std::tuple_element_t<1, OutParameters>  recv_counts_status;
        std::tuple_element_t<2, OutParameters>  recv_displs_status;
        std::tuple_element_t<3, OutParameters>  send_counts_status;
        std::tuple_element_t<4, OutParameters>  send_displs_status;
        std::tuple_element_t<5, OutParameters>  send_count;
        std::tuple_element_t<6, OutParameters>  recv_count;
        std::tuple_element_t<7, OutParameters>  send_recv_count;
        std::tuple_element_t<8, OutParameters>  send_type;
        std::tuple_element_t<9, OutParameters>  recv_type;
        std::tuple_element_t<10, OutParameters> send_recv_type;
        auto                                    result_status = make_mpi_result<OutParameters>(
            std::move(recv_counts_status),
            std::move(recv_count),
            std::move(recv_displs_status),
            std::move(send_counts_status),
            std::move(send_count),
            std::move(send_displs_status),
            std::move(recv_buf_status),
            std::move(send_recv_count),
            std::move(send_type),
            std::move(recv_type),
            std::move(send_recv_type)
        );
        EXPECT_FALSE(has_member_extract_status_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_recv_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_type_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_type_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_recv_type_v<decltype(result_status)>);
        EXPECT_FALSE(decltype(result_status)::is_empty);
    }

    {
        using OutParameters = std::tuple<
            LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype>>;
        std::tuple_element_t<0, OutParameters> status_recv_counts;
        std::tuple_element_t<1, OutParameters> recv_buf_recv_counts;
        std::tuple_element_t<2, OutParameters> recv_displs_recv_counts;
        std::tuple_element_t<3, OutParameters> send_counts_recv_counts;
        std::tuple_element_t<4, OutParameters> send_displs_recv_counts;
        auto                                   result_recv_counts = make_mpi_result<OutParameters>(
            std::move(status_recv_counts),
            std::move(recv_buf_recv_counts),
            std::move(recv_displs_recv_counts),
            std::move(send_counts_recv_counts),
            std::move(send_displs_recv_counts)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_recv_counts)>);
        EXPECT_FALSE(has_member_extract_recv_counts_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_recv_counts)>);
        EXPECT_FALSE(decltype(result_recv_counts)::is_empty);
    }

    {
        using OutParameters = std::tuple<
            LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype>>;

        std::tuple_element_t<0, OutParameters> status_recv_displs;
        std::tuple_element_t<1, OutParameters> recv_buf_recv_displs;
        std::tuple_element_t<2, OutParameters> recv_counts_recv_displs;
        std::tuple_element_t<3, OutParameters> send_counts_recv_displs;
        std::tuple_element_t<4, OutParameters> send_displs_recv_displs;
        auto                                   result_recv_displs = make_mpi_result<OutParameters>(
            std::move(status_recv_displs),
            std::move(recv_buf_recv_displs),
            std::move(recv_counts_recv_displs),
            std::move(send_counts_recv_displs),
            std::move(send_displs_recv_displs)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_recv_displs)>);
        EXPECT_FALSE(has_member_extract_recv_displs_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_recv_displs)>);
        EXPECT_FALSE(decltype(result_recv_displs)::is_empty);
    }

    {
        using OutParameters = std::tuple<
            LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype>>;

        std::tuple_element_t<0, OutParameters> status_send_counts;
        std::tuple_element_t<1, OutParameters> recv_buf_send_counts;
        std::tuple_element_t<2, OutParameters> recv_counts_send_counts;
        std::tuple_element_t<3, OutParameters> recv_displs_send_counts;
        std::tuple_element_t<4, OutParameters> send_displs_send_counts;
        auto                                   result_send_counts = make_mpi_result<OutParameters>(
            std::move(status_send_counts),
            std::move(recv_buf_send_counts),
            std::move(recv_counts_send_counts),
            std::move(recv_displs_send_counts),
            std::move(send_displs_send_counts)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_send_counts)>);
        EXPECT_FALSE(has_member_extract_send_counts_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_send_counts)>);
        EXPECT_FALSE(decltype(result_send_counts)::is_empty);
    }

    {
        using OutParameters = std::tuple<
            LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype>>;
        std::tuple_element_t<0, OutParameters> status_send_displs;
        std::tuple_element_t<1, OutParameters> recv_buf_send_displs;
        std::tuple_element_t<2, OutParameters> recv_counts_send_displs;
        std::tuple_element_t<3, OutParameters> recv_displs_send_displs;
        std::tuple_element_t<4, OutParameters> send_counts_send_displs;
        auto                                   result_send_displs = make_mpi_result<OutParameters>(
            std::move(status_send_displs),
            std::move(recv_buf_send_displs),
            std::move(recv_counts_send_displs),
            std::move(recv_displs_send_displs),
            std::move(send_counts_send_displs)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_send_displs)>);
        EXPECT_FALSE(has_member_extract_send_displs_v<decltype(result_send_displs)>);
        EXPECT_FALSE(decltype(result_send_displs)::is_empty);
    }
}

TEST(MakeMpiResultTest, structured_bindings_basics) {
    constexpr BufferType btype = BufferType::out_buffer;
    {
        // structured binding by value
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<std::int8_t>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int16_t>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int32_t>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int64_t>, ParameterType::send_counts, btype>>;
        std::tuple_element_t<0, OutParameters> recv_buf;
        std::tuple_element_t<1, OutParameters> recv_counts_buf;
        std::tuple_element_t<2, OutParameters> recv_displs_buf;
        std::tuple_element_t<3, OutParameters> send_counts_buf;
        auto [recv_buffer, recv_counts, recv_displs, send_counts] = make_mpi_result<OutParameters>(
            std::move(recv_buf),
            std::move(recv_counts_buf),
            std::move(recv_displs_buf),
            std::move(send_counts_buf)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_buffer)>, std::vector<std::int8_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_counts)>, std::vector<std::int16_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_displs)>, std::vector<std::int32_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(send_counts)>, std::vector<std::int64_t>>);
    }
    {
        // structured binding by rvalue ref
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<std::int8_t>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int16_t>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int32_t>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int64_t>, ParameterType::send_counts, btype>>;
        std::tuple_element_t<0, OutParameters> recv_buf;
        std::tuple_element_t<1, OutParameters> recv_counts_buf;
        std::tuple_element_t<2, OutParameters> recv_displs_buf;
        std::tuple_element_t<3, OutParameters> send_counts_buf;
        auto&& [recv_buffer, recv_counts, recv_displs, send_counts] = make_mpi_result<OutParameters>(
            std::move(recv_buf),
            std::move(recv_counts_buf),
            std::move(recv_displs_buf),
            std::move(send_counts_buf)
        );
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_buffer)>, std::vector<std::int8_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_counts)>, std::vector<std::int16_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_displs)>, std::vector<std::int32_t>>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(send_counts)>, std::vector<std::int64_t>>);
    }
    {
        // structured binding by const value
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<std::int8_t>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int16_t>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int32_t>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int64_t>, ParameterType::send_counts, btype>>;
        std::tuple_element_t<0, OutParameters> recv_buf;
        std::tuple_element_t<1, OutParameters> recv_counts_buf;
        std::tuple_element_t<2, OutParameters> recv_displs_buf;
        std::tuple_element_t<3, OutParameters> send_counts_buf;
        auto const [recv_buffer, recv_counts, recv_displs, send_counts] = make_mpi_result<OutParameters>(
            std::move(recv_buf),
            std::move(recv_counts_buf),
            std::move(recv_displs_buf),
            std::move(send_counts_buf)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_buffer)>, std::vector<std::int8_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_counts)>, std::vector<std::int16_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_displs)>, std::vector<std::int32_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(send_counts)>, std::vector<std::int64_t> const>);
    }
    {
        // structured binding by const reference
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<std::int8_t>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int16_t>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int32_t>, ParameterType::recv_displs, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int64_t>, ParameterType::send_counts, btype>>;
        std::tuple_element_t<0, OutParameters> recv_buf;
        std::tuple_element_t<1, OutParameters> recv_counts_buf;
        std::tuple_element_t<2, OutParameters> recv_displs_buf;
        std::tuple_element_t<3, OutParameters> send_counts_buf;
        auto const& [recv_buffer, recv_counts, recv_displs, send_counts] = make_mpi_result<OutParameters>(
            std::move(recv_buf),
            std::move(recv_counts_buf),
            std::move(recv_displs_buf),
            std::move(send_counts_buf)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_buffer)>, std::vector<std::int8_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_counts)>, std::vector<std::int16_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(recv_displs)>, std::vector<std::int32_t> const>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(send_counts)>, std::vector<std::int64_t> const>);
    }
}

TEST(MakeMpiResultTest, pass_random_order_buffer) {
    {
        constexpr BufferType btype = BufferType::out_buffer;
        using OutParameters        = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<std::int8_t>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<std::int32_t>, ParameterType::recv_displs, btype>,
            LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>>;
        std::tuple_element_t<0, OutParameters> recv_counts;
        std::tuple_element_t<1, OutParameters> recv_buf;
        std::tuple_element_t<2, OutParameters> recv_displs;
        std::tuple_element_t<3, OutParameters> status;
        status_param_to_native_ptr(status)->MPI_TAG = 42;

        auto result = make_mpi_result<OutParameters>(
            std::move(recv_counts),
            std::move(status),
            std::move(recv_buf),
            std::move(recv_displs)
        );

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();
        auto result_recv_displs = result.extract_recv_displs();
        auto result_status      = result.extract_status();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, char>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int8_t>);
        static_assert(std::is_same_v<decltype(result_recv_displs)::value_type, int32_t>);
        ASSERT_EQ(result_status.tag(), 42);
    }
    {
        constexpr BufferType btype = BufferType::out_buffer;
        using OutParameters        = std::tuple<
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>,
            LibAllocatedContainerBasedBuffer<std::vector<double>, ParameterType::recv_buf, btype>>;

        std::tuple_element_t<0, OutParameters> recv_counts;
        std::tuple_element_t<1, OutParameters> recv_buf;

        auto result = make_mpi_result<OutParameters>(std::move(recv_counts), std::move(recv_buf));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, double>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int>);
    }
}

TEST(MakeMpiResultTest, pass_send_recv_buf) {
    {
        using T =
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_recv_buf, BufferType::in_out_buffer>;
        T    send_recv_buf;
        auto result_recv_buf = make_mpi_result<T>(std::move(send_recv_buf));
        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, int>);
    }
}

TEST(MakeMpiResultTest, pass_send_recv_buf_and_other_out_parameters) {
    {
        using OutParameters = std::tuple<
            LibAllocatedContainerBasedBuffer<
                std::vector<char>,
                ParameterType::send_recv_buf,
                BufferType::in_out_buffer>,
            LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, BufferType::out_buffer>>;

        std::tuple_element_t<0, OutParameters> send_recv_buf;
        std::tuple_element_t<1, OutParameters> send_counts;
        auto result = make_mpi_result<OutParameters>(std::move(send_recv_buf), std::move(send_counts));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_send_counts = result.extract_send_counts();
        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, char>);
        static_assert(std::is_same_v<decltype(result_send_counts)::value_type, int>);
    }
}

TEST(MakeMpiResultTest, pass_send_recv_buf_and_other_out_parameters_as_structured_bindings) {
    using OutParameters = std::tuple<
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::send_recv_buf, BufferType::in_out_buffer>,
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, BufferType::out_buffer>>;

    std::tuple_element_t<0, OutParameters> send_recv_buf;
    std::tuple_element_t<1, OutParameters> send_counts;

    {
        auto [result_recv_buf, result_send_counts] =
            make_mpi_result<OutParameters>(std::move(send_counts), std::move(send_recv_buf));

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_send_counts)>::value_type, int>);
    }
    {
        // send_recv_buf not given
        using ImplicitSendRecvBuf = std::tuple<std::tuple_element_t<1, OutParameters>>;
        auto [result_recv_buf, result_send_counts] =
            make_mpi_result<ImplicitSendRecvBuf>(std::move(send_counts), std::move(send_recv_buf));

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_send_counts)>::value_type, int>);
    }
}

TEST(MakeMpiResultTest, check_content) {
    constexpr BufferType btype = BufferType::out_buffer;

    using OutParameters = std::tuple<
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_buf, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_counts, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_displs, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::send_displs, btype>>;

    std::vector<int> recv_buf_data(20);
    std::iota(recv_buf_data.begin(), recv_buf_data.end(), 0);
    Span<int>                              recv_buf_container = {recv_buf_data.data(), recv_buf_data.size()};
    std::tuple_element_t<0, OutParameters> recv_buf(recv_buf_container);

    std::vector<int> recv_counts_data(20);
    std::iota(recv_counts_data.begin(), recv_counts_data.end(), 20);
    Span<int>                              recv_counts_container = {recv_counts_data.data(), recv_counts_data.size()};
    std::tuple_element_t<1, OutParameters> recv_counts(recv_counts_container);

    std::vector<int> recv_displs_data(20);
    std::iota(recv_displs_data.begin(), recv_displs_data.end(), 40);
    Span<int>                              recv_displs_container = {recv_displs_data.data(), recv_displs_data.size()};
    std::tuple_element_t<2, OutParameters> recv_displs(recv_displs_container);

    std::vector<int> send_displs_data(20);
    std::iota(send_displs_data.begin(), send_displs_data.end(), 60);
    Span<int>                              send_displs_container = {send_displs_data.data(), send_displs_data.size()};
    std::tuple_element_t<3, OutParameters> send_displs(send_displs_container);

    auto result = make_mpi_result<OutParameters>(
        std::move(recv_buf),
        std::move(recv_counts),
        std::move(recv_displs),
        std::move(send_displs)
    );

    auto result_recv_buf = result.extract_recv_buffer();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_buf.data()[i], i);
    }
    auto result_recv_counts = result.extract_recv_counts();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_counts.data()[i], i + 20);
    }
    auto result_recv_displs = result.extract_recv_displs();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_displs.data()[i], i + 40);
    }
    auto result_send_displs = result.extract_send_displs();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_send_displs.data()[i], i + 60);
    }
}

TEST(MakeMpiResultTest, check_content_structured_binding) {
    constexpr BufferType btype = BufferType::out_buffer;

    using OutParameters = std::tuple<
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_buf, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_counts, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_displs, btype>,
        LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::send_displs, btype>>;

    std::vector<int> recv_buf_data(20);
    std::iota(recv_buf_data.begin(), recv_buf_data.end(), 0);
    Span<int>                              recv_buf_container = {recv_buf_data.data(), recv_buf_data.size()};
    std::tuple_element_t<0, OutParameters> recv_buf(recv_buf_container);

    std::vector<int> recv_counts_data(20);
    std::iota(recv_counts_data.begin(), recv_counts_data.end(), 20);
    Span<int>                              recv_counts_container = {recv_counts_data.data(), recv_counts_data.size()};
    std::tuple_element_t<1, OutParameters> recv_counts(recv_counts_container);

    std::vector<int> recv_displs_data(20);
    std::iota(recv_displs_data.begin(), recv_displs_data.end(), 40);
    Span<int>                              recv_displs_container = {recv_displs_data.data(), recv_displs_data.size()};
    std::tuple_element_t<2, OutParameters> recv_displs(recv_displs_container);

    std::vector<int> send_displs_data(20);
    std::iota(send_displs_data.begin(), send_displs_data.end(), 60);
    Span<int>                              send_displs_container = {send_displs_data.data(), send_displs_data.size()};
    std::tuple_element_t<3, OutParameters> send_displs(send_displs_container);

    auto [result_recv_buf, result_recv_counts, result_recv_displs, result_send_displs] = make_mpi_result<OutParameters>(
        std::move(recv_buf),
        std::move(recv_counts),
        std::move(recv_displs),
        std::move(send_displs)
    );

    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_buf.data()[i], i);
    }
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_counts.data()[i], i + 20);
    }
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_displs.data()[i], i + 40);
    }
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_send_displs.data()[i], i + 60);
    }
}

TEST(MakeMpiResultTest, check_handling_of_recv_buffer) {
    constexpr BufferType btype = BufferType::out_buffer;

    using OwningOutRecvBuf    = LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>;
    using OwningOutRecvCounts = LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>;
    using NonOwningOutRecvCounts =
        UserAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype, no_resize>;

    std::vector<int> non_owning_recv_counts_storage;

    {
        // no caller provided owning out buffers
        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto result_recv_buf = make_mpi_result<std::tuple<>>(std::move(recv_buf), std::move(recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    {
        // no caller provided owning recv buffer with non-owning other out parameter
        OwningOutRecvBuf       recv_buf;
        NonOwningOutRecvCounts non_owning_recv_counts(non_owning_recv_counts_storage);
        auto                   result_recv_buf =
            make_mpi_result<std::tuple<NonOwningOutRecvCounts>>(std::move(recv_buf), std::move(non_owning_recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    { // no caller provided recv buffer with other owning out parameter
        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto result = make_mpi_result<std::tuple<OwningOutRecvCounts>>(std::move(recv_buf), std::move(recv_counts));
        static_assert(std::
                          is_same_v<std::remove_reference_t<decltype(result.extract_recv_buffer())>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result.extract_recv_counts())>::value_type, int>);
    }
    {
        // no caller provided recv buffer without other owning out parameters

        OwningOutRecvBuf       recv_buf;
        NonOwningOutRecvCounts non_owning_recv_counts(non_owning_recv_counts_storage);
        auto                   result_recv_buf =
            make_mpi_result<std::tuple<NonOwningOutRecvCounts>>(std::move(recv_buf), std::move(non_owning_recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    { // caller provided owning recv counts and recv buf - changed order!

        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto                result = make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutRecvBuf>>(
            std::move(recv_buf),
            std::move(recv_counts)
        );
        static_assert(std::
                          is_same_v<std::remove_reference_t<decltype(result.extract_recv_buffer())>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result.extract_recv_counts())>::value_type, int>);
    }
    {
        // caller provided owning recv counts and recv buf - changed order!

        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto                result = make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutRecvBuf>>(
            std::move(recv_buf),
            std::move(recv_counts)
        );

        static_assert(std::
                          is_same_v<std::remove_reference_t<decltype(result.extract_recv_buffer())>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result.extract_recv_counts())>::value_type, int>);
    }
}

TEST(MakeMpiResultTest, check_order_of_handling_of_recv_buffer) {
    constexpr BufferType btype = BufferType::out_buffer;

    using OwningOutRecvBuf    = LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>;
    using OwningOutRecvCounts = LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>;

    {
        // no caller provided recv buffer with other owning out parameter
        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto [result_recv_buf, result_recv_counts] =
            make_mpi_result<std::tuple<OwningOutRecvCounts>>(std::move(recv_buf), std::move(recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
    {
        // caller provided owning recv counts and recv buf - changed order!

        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto [result_recv_counts, result_recv_buf] = make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutRecvBuf>>(
            std::move(recv_buf),
            std::move(recv_counts)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
    {
        // caller provided owning recv counts and recv buf - changed order!

        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto [result_recv_counts, result_recv_buf] = make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutRecvBuf>>(
            std::move(recv_buf),
            std::move(recv_counts)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
    {
        // caller provided owning recv counts and recv buf - changed order!

        OwningOutRecvBuf    recv_buf;
        OwningOutRecvCounts recv_counts;
        auto [result_recv_counts, result_recv_buf] = make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutRecvBuf>>(
            std::move(recv_buf),
            std::move(recv_counts)
        );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
}

TEST(MakeMpiResultTest, check_order_of_handling_of_send_recv_buffer) {
    constexpr BufferType btype = BufferType::out_buffer;

    using OwningOutSendRecvBuf =
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::send_recv_buf, btype>;
    using OwningOutRecvCounts = LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype>;
    using NonOwningOutRecvCounts =
        UserAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype, no_resize>;

    std::vector<int> non_owning_recv_counts_storage;

    {
        // no caller provided owning out buffers
        OwningOutSendRecvBuf send_recv_buf;
        OwningOutRecvCounts  recv_counts;

        auto result_recv_buf = make_mpi_result<std::tuple<>>(std::move(send_recv_buf), std::move(recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    {
        // no caller provided owning send_recv buffer with non-owning other out parameter
        OwningOutSendRecvBuf   send_recv_buf;
        NonOwningOutRecvCounts non_owning_recv_counts(non_owning_recv_counts_storage);
        auto                   result_recv_buf = make_mpi_result<std::tuple<NonOwningOutRecvCounts>>(
            std::move(send_recv_buf),
            std::move(non_owning_recv_counts)
        );
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    {
        // no caller provided send_recv buffer with other owning out parameter
        OwningOutSendRecvBuf send_recv_buf;
        OwningOutRecvCounts  recv_counts;
        auto [result_recv_buf, result_recv_counts] =
            make_mpi_result<std::tuple<OwningOutRecvCounts>>(std::move(send_recv_buf), std::move(recv_counts));
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
    {
        // no caller provided send_recv buffer without other owning out parameters
        OwningOutSendRecvBuf   send_recv_buf;
        NonOwningOutRecvCounts non_owning_recv_counts(non_owning_recv_counts_storage);
        auto                   result_recv_buf = make_mpi_result<std::tuple<NonOwningOutRecvCounts>>(
            std::move(send_recv_buf),
            std::move(non_owning_recv_counts)
        );
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
    }
    {
        // caller provided owning send_recv counts and recv buf - changed order!
        OwningOutSendRecvBuf send_recv_buf;
        OwningOutRecvCounts  recv_counts;
        auto [result_recv_counts, result_recv_buf] =
            make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutSendRecvBuf>>(
                std::move(send_recv_buf),
                std::move(recv_counts)
            );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
    {
        // caller provided owning send_recv counts and recv buf - changed order!
        OwningOutSendRecvBuf send_recv_buf;
        OwningOutRecvCounts  recv_counts;
        auto [result_recv_counts, result_recv_buf] =
            make_mpi_result<std::tuple<OwningOutRecvCounts, OwningOutSendRecvBuf>>(
                std::move(send_recv_buf),
                std::move(recv_counts)
            );

        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_buf)>::value_type, char>);
        static_assert(std::is_same_v<std::remove_reference_t<decltype(result_recv_counts)>::value_type, int>);
    }
}

template <template <typename> typename Container>
auto construct_mpi_result_object_with_recv_counts_and_recv_buf(
    Container<int>&& recv_counts_container, Container<char>&& recv_buffer_container
) {
    using RecvCountsType =
        LibAllocatedContainerBasedBuffer<Container<int>, ParameterType::recv_counts, BufferType::out_buffer>;
    using RecvBufType =
        LibAllocatedContainerBasedBuffer<Container<char>, ParameterType::recv_buf, BufferType::out_buffer>;

    MPIResult<RecvCountsType, RecvBufType> result(
        std::make_tuple(std::move(recv_counts_container), std::move(recv_buffer_container))
    );
    return result;
}

TEST(MpiResultTest, structured_bindings_with_copy_counting_containers_by_value) {
    testing::OwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::OwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );
    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_EQ(recv_counts.copy_count(), 0);
    EXPECT_EQ(recv_buf.copy_count(), 0);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_copy_counting_containers_by_const_value) {
    testing::OwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::OwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto const [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(std::is_const_v<decltype(recv_counts)>);
    static_assert(std::is_const_v<decltype(recv_buf)>);
    EXPECT_EQ(recv_counts.copy_count(), 0);
    EXPECT_EQ(recv_buf.copy_count(), 0);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_copy_counting_containers_by_with_lvalue_ref) {
    testing::OwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::OwnContainer<char> recv_buf_container{3, 4, 5, 6};
    auto                        result = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );
    auto& [recv_counts, recv_buf] = result;

    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_EQ(recv_counts.copy_count(), 0);
    EXPECT_EQ(recv_buf.copy_count(), 0);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_copy_counting_containers_by_const_lvalue_ref) {
    testing::OwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::OwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto const& [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(std::is_const_v<decltype(recv_counts)>);
    static_assert(std::is_const_v<decltype(recv_buf)>);
    EXPECT_EQ(recv_counts.copy_count(), 0);
    EXPECT_EQ(recv_buf.copy_count(), 0);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_copy_counting_containers_by_rvalue_ref) {
    testing::OwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::OwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto&& [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_EQ(recv_counts.copy_count(), 0);
    EXPECT_EQ(recv_buf.copy_count(), 0);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_non_copyable_containers_by_value) {
    testing::NonCopyableOwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::NonCopyableOwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_non_copyable_containers_by_const_value) {
    testing::NonCopyableOwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::NonCopyableOwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto const [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(std::is_const_v<decltype(recv_counts)>);
    static_assert(std::is_const_v<decltype(recv_buf)>);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_lvalue_ref) {
    testing::NonCopyableOwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::NonCopyableOwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto result = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );
    auto& [recv_counts, recv_buf] = result;

    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_const_lvalue_ref) {
    testing::NonCopyableOwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::NonCopyableOwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto const& [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(std::is_const_v<decltype(recv_counts)>);
    static_assert(std::is_const_v<decltype(recv_buf)>);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}

TEST(MpiResultTest, structured_bindings_with_non_copyable_containers_rvalue_ref) {
    testing::NonCopyableOwnContainer<int>  recv_counts_container{0, 1, 2};
    testing::NonCopyableOwnContainer<char> recv_buf_container{3, 4, 5, 6};

    auto&& [recv_counts, recv_buf] = construct_mpi_result_object_with_recv_counts_and_recv_buf(
        std::move(recv_counts_container),
        std::move(recv_buf_container)
    );

    static_assert(!std::is_const_v<decltype(recv_counts)>);
    static_assert(!std::is_const_v<decltype(recv_buf)>);
    EXPECT_THAT(recv_counts, testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(recv_buf, testing::ElementsAre(3, 4, 5, 6));
}
