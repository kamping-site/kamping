#pragma once
#include <memory>
#include <vector>

#include <mpi.h>

#include "data_buffer.hpp"
#include "kamping/status.hpp"

namespace kamping {
template <typename Allocator = std::allocator<MPI_Status>>
class status_vector : public std::vector<MPI_Status> {
public:
    using value_type      = Status;
    using reference       = value_type&;
    using const_reference = value_type const&;
};

template <typename Allocator>
class internal::ValueTypeWrapper</*has_value_type_member =*/true, status_vector<Allocator>> {
public:
    using value_type = MPI_Status; ///< The value type of T.
};
} // namespace kamping
