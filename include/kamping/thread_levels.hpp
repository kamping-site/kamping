#pragma once

#include <optional>

#include <kamping/info.hpp>
#include <mpi.h>

namespace kamping {
enum class ThreadLevel : int {
    single     = MPI_THREAD_SINGLE,
    funneled   = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple   = MPI_THREAD_MULTIPLE
};

template <>
inline std::string_view to_info_value_string<ThreadLevel>(ThreadLevel const& value) {
    switch (value) {
        case ThreadLevel::single:
            return "MPI_THREAD_SINGLE";
        case ThreadLevel::funneled:
            return "MPI_THREAD_FUNNELED";
        case ThreadLevel::serialized:
            return "MPI_THREAD_SERIALIZED";
        case ThreadLevel::multiple:
            return "MPI_THREAD_MULTIPLE";
    }
}

template <>
inline std::optional<ThreadLevel> from_info_value_string<ThreadLevel>(std::string_view value) {
    if (value == "MPI_THREAD_SINGLE") {
        return ThreadLevel::single;
    }
    if (value == "MPI_THREAD_FUNNELED") {
        return ThreadLevel::funneled;
    }
    if (value == "MPI_THREAD_SERIALIZED") {
        return ThreadLevel::serialized;
    }
    if (value == "MPI_THREAD_MULTIPLE") {
        return ThreadLevel::multiple;
    }
    return std::nullopt;
}
} // namespace kamping
