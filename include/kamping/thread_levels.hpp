// This file is part of KaMPIng.
//
// Copyright 2022-2026 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief MPI thread support levels.

#pragma once

#include <mpi.h>

namespace kamping {
/// @brief MPI thread support levels defining the allowed concurrency of MPI calls relative to application threads.
/// You can obtain the underlying values by casting the enum value to \c int.
enum class ThreadLevel : int {
    /// No thread support; only one thread may execute, and only the main thread can make MPI calls.
    single = MPI_THREAD_SINGLE,
    /// Only the main thread will make MPI calls, but the application may be multi-threaded.
    funneled = MPI_THREAD_FUNNELED,
    /// Multiple threads may exist, but only one at a time will make MPI calls (calls are serialized).
    serialized = MPI_THREAD_SERIALIZED,
    /// Full thread support; multiple threads may call MPI concurrently.
    multiple = MPI_THREAD_MULTIPLE
};

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator==(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) == static_cast<int>(rhs);
}

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator!=(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return !(lhs == rhs);
}

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator<(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) < static_cast<int>(rhs);
}

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator<=(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) <= static_cast<int>(rhs);
}

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator>(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) > static_cast<int>(rhs);
}

/// @brief Comparison operator for \ref ThreadLevel.
inline bool operator>=(ThreadLevel lhs, ThreadLevel rhs) noexcept {
    return static_cast<int>(lhs) >= static_cast<int>(rhs);
}
} // namespace kamping
