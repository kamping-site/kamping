#include <limits>
// somehow in combination with GLIBCXX_DEBUG, this does not compile if the
// include order is the over way around
// clang-format off
#include <vector>
#include <unordered_map>
// clang-format on

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/allocator.hpp"

class AllocatorTest : public ::testing::Test {
public:
    static inline size_t                            allocated_memory = 0;
    static inline std::unordered_map<void*, size_t> chunks;

    void SetUp() override {
        allocated_memory = 0;
    }

    void TearDown() override {
        allocated_memory = 0;
    }
};

// we keep track of the allocated memory ourselves

int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void* baseptr) {
    int err = PMPI_Alloc_mem(size, info, baseptr);
    if (err == MPI_SUCCESS) {
        AllocatorTest::allocated_memory += static_cast<size_t>(size);
        AllocatorTest::chunks[*static_cast<void**>(baseptr)] = static_cast<size_t>(size);
    }
    return err;
}

int MPI_Free_mem(void* base) {
    auto it = AllocatorTest::chunks.find(base);
    if (it == AllocatorTest::chunks.end()) {
        // we cannot assert here, because this is only allow inside test macros
        throw std::runtime_error("Chunk not previously allocated");
    }
    int err = PMPI_Free_mem(base);
    if (err == MPI_SUCCESS) {
        AllocatorTest::allocated_memory -= it->second;
        AllocatorTest::chunks.erase(it);
    }
    return err;
}

TEST_F(AllocatorTest, simple_allocation) {
    {
        kamping::MPIAllocator<std::byte> alloc;
        std::byte*                       ptr1 = alloc.allocate(42);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(std::byte) * 42);
        std::byte* ptr2 = alloc.allocate(1);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(std::byte) * 43);
        alloc.deallocate(ptr1, 42);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(std::byte) * 1);
        alloc.deallocate(ptr2, 1);
        EXPECT_EQ(AllocatorTest::allocated_memory, 0);
        EXPECT_TRUE(AllocatorTest::chunks.empty());
    }
    {
        kamping::MPIAllocator<double> alloc;
        double*                       ptr1 = alloc.allocate(42);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(double) * 42);
        double* ptr2 = alloc.allocate(1);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(double) * 43);
        alloc.deallocate(ptr1, 42);
        EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(double) * 1);
        alloc.deallocate(ptr2, 1);
        EXPECT_EQ(AllocatorTest::allocated_memory, 0);
        EXPECT_TRUE(AllocatorTest::chunks.empty());
    }
}

struct MyType {
    std::byte data[32];
};

TEST_F(AllocatorTest, custom_type_allocation) {
    kamping::MPIAllocator<double>                                       double_alloc;
    std::allocator_traits<decltype(double_alloc)>::rebind_alloc<MyType> my_type_alloc;
    MyType*                                                             ptr = my_type_alloc.allocate(1);
    EXPECT_EQ(sizeof(MyType), sizeof(std::byte) * 32);
    EXPECT_EQ(AllocatorTest::allocated_memory, sizeof(MyType));
    my_type_alloc.deallocate(ptr, 1);
    EXPECT_EQ(AllocatorTest::allocated_memory, 0);
}

TEST_F(AllocatorTest, vector_allocation) {
    std::vector<double, kamping::MPIAllocator<double>> v;
    EXPECT_EQ(AllocatorTest::allocated_memory, 0);
    v.push_back(4.12);
    v.push_back(135.134);
    v.push_back(351.123);
    v.push_back(0);
    EXPECT_EQ(v.capacity() * sizeof(double), AllocatorTest::allocated_memory);
    v.shrink_to_fit();
    EXPECT_EQ(v.capacity() * sizeof(double), AllocatorTest::allocated_memory);
    std::vector<double, kamping::MPIAllocator<double>>{}.swap(v);
    EXPECT_EQ(v.capacity(), 0);
    EXPECT_EQ(AllocatorTest::allocated_memory, 0);
}

// we cannot really test for the case if memory allocation fails, only for the
// case of exceeding the bounds of MPI's MPI_Aint
TEST_F(AllocatorTest, size_out_of_bound) {
    kamping::MPIAllocator<std::byte> alloc;
    EXPECT_THROW(alloc.allocate(static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()) + 1), std::runtime_error);
    EXPECT_EQ(AllocatorTest::allocated_memory, 0);
}
