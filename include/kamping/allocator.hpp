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

#pragma once

#include <cstddef>
#include <limits>
#include <memory>

#include <mpi.h>

#include "kamping/error_handling.hpp"

namespace kamping {

/// @brief STL-compatible allocator for requesting memory using the builtin MPI allocator.
///
/// Note that this allocator may only be used after initializing MPI.
///
/// @tparam T The type to allocate.
template <typename T>
class MPIAllocator {
public:
    // Note: this implements all required functionality of a custom allocator,
    // the rest is inferred.
    //
    // This (almost minimal) set of implemented members for the named
    // requirement `Allocator` matches the ones of Intel's TBB allocator, which
    // works similar to ours.
    //
    // See https://en.cppreference.com/w/cpp/named_req/Allocator for details.

    MPIAllocator() noexcept = default;

    /// @brief Copy constructor for allocators with different value type.
    ///
    /// Since the allocator is stateless, we can also copy-assign
    /// allocators for other types (because this is a noop).
    template <typename U>
    MPIAllocator(MPIAllocator<U> const&) noexcept {}

    /// @brief The value type.
    using value_type = T;

    /// @brief the memory "ownership" can be moved when the container is
    /// move-assigned. If this would not be the case, container would need to
    /// free memory using the old allocator and reallocated it using the copied
    /// allocator.
    using propagate_on_container_move_assignment = std::true_type;

    /// @brief memory allocated by one allocator instance can always be dellocated
    /// by another and vice-versa
    using is_always_equal = std::true_type;

    /// @brief Allocates <tt> n * sizeof(T) </tt> bytes using MPI allocation functions.
    /// @param n The number of objects to allocate storage for.
    /// @return Pointer to the allocated memory segment.
    T* allocate(size_t n) {
        T* ptr;
        if (sizeof(value_type) * n > std::numeric_limits<MPI_Aint>::max()) {
            throw std::runtime_error("Requested allocation exceeds MPI address size.");
        }
        MPI_Aint alloc_size = static_cast<MPI_Aint>(sizeof(value_type) * n);
        int      err        = MPI_Alloc_mem(alloc_size, MPI_INFO_NULL, &ptr);
        THROW_IF_MPI_ERROR(err, MPI_Alloc_mem);
        return ptr;
    }

    /// @brief Deallocates the storage referenced by the pointer \c p, which must be a pointer obtained by an earlier
    /// call to \ref allocate().
    /// @param p Pointer obtained from \ref allocate().
    void deallocate(T* p, size_t) {
        // no error handling because the standard disallows throwing exceptions here
        MPI_Free_mem(p);
    }
};

// From https://en.cppreference.com/w/cpp/named_req/Allocator:
// - true only if the storage allocated by the allocator a1 can be deallocated through a2.
// - Establishes reflexive, symmetric, and transitive relationship.
// - Does not throw exceptions.
template <typename T, typename U>
bool operator==(MPIAllocator<T> const&, MPIAllocator<U> const&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(MPIAllocator<T> const&, MPIAllocator<U> const&) noexcept {
    return false;
}
} // namespace kamping
