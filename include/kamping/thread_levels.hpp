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
struct info_value_traits<ThreadLevel> {
    using type = ThreadLevel;
    static std::string_view to(ThreadLevel value) {
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

    static std::optional<ThreadLevel> from(std::string_view value) {
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
};
} // namespace kamping
