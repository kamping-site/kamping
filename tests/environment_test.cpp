// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "test_assertions.hpp"

#include <chrono>
#include <set>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/environment.hpp"

using namespace ::kamping;

std::set<MPI_Datatype> freed_types;
void*                  attached_buffer_ptr  = nullptr;
int                    attached_buffer_size = 0;
void*                  detached_buffer_ptr  = nullptr;
int                    detached_buffer_size = 0;

int MPI_Type_free(MPI_Datatype* type) {
    freed_types.insert(*type);
    return PMPI_Type_free(type);
}

struct EnvironmentTest : ::testing::Test {
    void SetUp() override {
        int  flag;
        int* value;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
        EXPECT_TRUE(flag);
        mpi_tag_ub = *value;
        freed_types.clear();
        attached_buffer_ptr  = nullptr;
        attached_buffer_size = 0;
        detached_buffer_ptr  = nullptr;
        detached_buffer_size = 0;
    }

    void TearDown() override {
        void* buffer;
        freed_types.clear();
        int size;
        MPI_Buffer_detach(&buffer, &size);
        attached_buffer_ptr  = nullptr;
        attached_buffer_size = 0;
        detached_buffer_ptr  = nullptr;
        detached_buffer_size = 0;
    }

    int mpi_tag_ub;
};

TEST_F(EnvironmentTest, wtime) {
    std::chrono::milliseconds::rep const milliseconds_to_sleep = 10;
    double const                         seconds_to_sleep      = static_cast<double>(milliseconds_to_sleep) / 1000.0;
    // Get the first time from an object
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    double                                            start_time = env.wtime();

    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds_to_sleep));

    // Get the second time from the class to check that wtime is static
    double end_time = Environment<>::wtime();

    EXPECT_GE(end_time, start_time + seconds_to_sleep);
}

TEST_F(EnvironmentTest, wtick) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    double                                            kamping_wtick = env.wtick();
    EXPECT_DOUBLE_EQ(kamping_wtick, MPI_Wtick());

    kamping_wtick = Environment<>::wtick();
    EXPECT_DOUBLE_EQ(kamping_wtick, MPI_Wtick());
}

TEST_F(EnvironmentTest, init) {
    // MPI_Init was already called by our custom test main().
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(env.initialized());
    // This should succeed because init checks whether MPI_Init has already been called.
    env.init();
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(EnvironmentTest, init_unchecked) {
    // MPI_Init was already called by our custom test main().
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(env.initialized());
    EXPECT_KASSERT_FAILS(env.init_unchecked(), "Trying to call MPI_Init twice");
}
#endif

TEST_F(EnvironmentTest, tag_upper_bound) {
    EXPECT_EQ(mpi_env.tag_upper_bound(), mpi_tag_ub);
    EXPECT_GE(mpi_env.tag_upper_bound(), 32767); // the standard requires that MPI_TAG_UB has at least this size
}

TEST_F(EnvironmentTest, is_valid_tag) {
    EXPECT_TRUE(mpi_env.is_valid_tag(0));
    EXPECT_TRUE(mpi_env.is_valid_tag(42));
    EXPECT_TRUE(mpi_env.is_valid_tag(mpi_tag_ub));
    // Avoid signed integer overflow
    if (mpi_tag_ub < std::numeric_limits<decltype(mpi_tag_ub)>::max()) {
        EXPECT_FALSE(mpi_env.is_valid_tag(mpi_tag_ub + 1));
    }

    if (mpi_tag_ub == std::numeric_limits<int>::max()) {
        EXPECT_TRUE(mpi_env.is_valid_tag(std::numeric_limits<int>::max()));
    } else {
        EXPECT_FALSE(mpi_env.is_valid_tag(std::numeric_limits<int>::max()));
    }
    EXPECT_FALSE(mpi_env.is_valid_tag(-1));
    EXPECT_FALSE(mpi_env.is_valid_tag(-42));
    EXPECT_FALSE(mpi_env.is_valid_tag(std::numeric_limits<int>::min()));
}

static MPI_Datatype last_committed_type = MPI_DATATYPE_NULL;

int MPI_Type_commit(MPI_Datatype* type) {
    last_committed_type = *type;
    return PMPI_Type_commit(type);
}

TEST_F(EnvironmentTest, commit_test) {
    last_committed_type = MPI_DATATYPE_NULL;
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(freed_types.empty());
    MPI_Datatype type;
    MPI_Type_contiguous(1, MPI_CHAR, &type);
    EXPECT_EQ(last_committed_type, MPI_DATATYPE_NULL);
    env.commit(type);
    EXPECT_EQ(last_committed_type, type);
    // nothing should have been registered
    EXPECT_TRUE(internal::registered_mpi_types.empty());
    MPI_Type_free(&type);
}

TEST_F(EnvironmentTest, free_test) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    MPI_Datatype                                      type;
    MPI_Type_contiguous(1, MPI_CHAR, &type);
    MPI_Type_commit(&type);
    EXPECT_TRUE(freed_types.empty());
    env.free(type);
    EXPECT_THAT(freed_types, ::testing::ElementsAre(type));
}

TEST_F(EnvironmentTest, commit_and_register_test) {
    last_committed_type = MPI_DATATYPE_NULL;
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(freed_types.empty());
    MPI_Datatype type;
    MPI_Type_contiguous(1, MPI_CHAR, &type);
    EXPECT_EQ(last_committed_type, MPI_DATATYPE_NULL);
    env.commit_and_register(type);
    EXPECT_EQ(last_committed_type, type);
    // the type should have been registered
    EXPECT_EQ(internal::registered_mpi_types.size(), 1);
    EXPECT_EQ(internal::registered_mpi_types.front(), type);
    env.free_registered_mpi_types();
}

TEST_F(EnvironmentTest, free_registered_tests) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    MPI_Datatype                                      type1, type2;
    MPI_Type_contiguous(1, MPI_CHAR, &type1);
    MPI_Type_commit(&type1);
    MPI_Type_contiguous(2, MPI_CHAR, &type2);
    MPI_Type_commit(&type2);

    MPI_Datatype type_null = MPI_DATATYPE_NULL;

    // Register with env object
    env.register_mpi_type(type1);
    env.register_mpi_type(type2);
    env.register_mpi_type(type_null);

    // Free with mpi_env object to test that the registered types are not bound to one object (or one template
    // specialization)
    mpi_env.free_registered_mpi_types();

    std::set<MPI_Datatype> expected_types({type1, type2});
    EXPECT_EQ(freed_types, expected_types);

    // Test that list of registered types is cleared after freeing them
    env.free_registered_mpi_types();
    freed_types.clear();
    EXPECT_TRUE(freed_types.empty());
}

int MPI_Buffer_attach(void* buffer, int size) {
    attached_buffer_ptr  = buffer;
    attached_buffer_size = size;
    return PMPI_Buffer_attach(buffer, size);
}

int MPI_Buffer_detach(void* buffer, int* size) {
    int err              = PMPI_Buffer_detach(buffer, size);
    detached_buffer_ptr  = *static_cast<void**>(buffer);
    detached_buffer_size = *size;
    return err;
}

TEST_F(EnvironmentTest, buffer_attach_and_detach) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    std::vector<int>                                  buffer;
    buffer.resize(42);
    env.buffer_attach(kamping::Span<int>{buffer.begin(), buffer.end()});

    EXPECT_EQ(attached_buffer_ptr, buffer.data());
    EXPECT_EQ(attached_buffer_size, 42 * sizeof(int));

    auto detached_buffer = env.buffer_detach<int>();

    EXPECT_EQ(detached_buffer_ptr, buffer.data());
    EXPECT_EQ(detached_buffer_size, 42 * sizeof(int));
    EXPECT_EQ(detached_buffer.data(), buffer.data());
    EXPECT_EQ(detached_buffer.size(), buffer.size());
}

TEST_F(EnvironmentTest, buffer_attach_and_detach_with_other_type) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    using attach_type = double;
    using detach_type = char;
    std::vector<attach_type> buffer;
    size_t                   buffer_size = std::max(size_t{13}, Environment<>::bsend_overhead);
    buffer.resize(buffer_size);
    env.buffer_attach(kamping::Span<attach_type>{buffer.begin(), buffer.end()});

    EXPECT_EQ(attached_buffer_ptr, buffer.data());
    EXPECT_EQ(attached_buffer_size, buffer_size * sizeof(attach_type));

    auto detached_buffer = env.buffer_detach<detach_type>();

    // test if the detached buffer machtes the original buffer
    EXPECT_EQ(attached_buffer_ptr, detached_buffer_ptr);
    EXPECT_EQ(attached_buffer_size, detached_buffer_size);

    EXPECT_EQ(detached_buffer_ptr, buffer.data());
    EXPECT_EQ(detached_buffer_size, buffer_size * sizeof(attach_type));
    EXPECT_EQ(static_cast<void*>(detached_buffer.data()), static_cast<void*>(buffer.data()));
    EXPECT_EQ(detached_buffer.size_bytes(), buffer_size * sizeof(attach_type));
}

TEST_F(EnvironmentTest, buffer_attach_and_detach_with_other_type_not_matching) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    using attach_type                  = char;
    using detach_type [[maybe_unused]] = double;
    std::vector<attach_type> buffer;
    size_t                   buffer_size = Environment<>::bsend_overhead + 1;
    buffer.resize(buffer_size);
    env.buffer_attach(kamping::Span<attach_type>{buffer.begin(), buffer.end()});

    EXPECT_EQ(attached_buffer_ptr, buffer.data());
    EXPECT_EQ(attached_buffer_size, buffer_size * sizeof(attach_type));

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(env.buffer_detach<detach_type>(), "The buffer size is not a multiple of the size of T.");
#endif
}

TEST_F(EnvironmentTest, buffer_attach_multiple_fails) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    std::vector<int>                                  buffer1;
    buffer1.resize(2 * Environment<>::bsend_overhead);
    std::vector<int> buffer2;
    buffer2.resize(Environment<>::bsend_overhead);
    env.buffer_attach(kamping::Span<int>{buffer1.begin(), buffer1.end()});
    EXPECT_EQ(attached_buffer_ptr, buffer1.data());
    EXPECT_EQ(attached_buffer_size, 2 * Environment<>::bsend_overhead * sizeof(int));

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(
        env.buffer_attach(kamping::Span<int>{buffer2.begin(), buffer2.end()}),
        "You may only attach one buffer at a time."
    );
#endif
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(EnvironmentTest, buffer_detach_none_fails) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_KASSERT_FAILS(env.buffer_detach<int>(), "There is currently no buffer attached.");
}
#endif

TEST_F(EnvironmentTest, buffer_detach_multiple_fails) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    std::vector<int>                                  buffer;
    size_t                                            buffer_size = std::max(size_t{42}, Environment<>::bsend_overhead);
    buffer.resize(buffer_size);
    env.buffer_attach(kamping::Span<int>{buffer.begin(), buffer.end()});
    EXPECT_EQ(attached_buffer_ptr, buffer.data());
    EXPECT_EQ(attached_buffer_size, buffer_size * sizeof(int));

    auto detached_buffer = env.buffer_detach<int>();
    EXPECT_EQ(detached_buffer_ptr, buffer.data());
    EXPECT_EQ(detached_buffer_size, buffer_size * sizeof(int));
    EXPECT_EQ(detached_buffer.data(), buffer.data());
    EXPECT_EQ(detached_buffer.size(), buffer.size());

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(env.buffer_detach<int>(), "There is currently no buffer attached.");
#endif
}
