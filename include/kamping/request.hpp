// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
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

#include <kamping/error_handling.hpp>
#include <mpi.h>

namespace kamping {

class Request {
public:
    Request(MPI_Request request = MPI_REQUEST_NULL) : _request(request) {}

    void wait() {
        int err = MPI_Wait(&_request, MPI_STATUS_IGNORE);
        THROW_IF_MPI_ERROR(err, MPI_Wait);
    }

    /// @return True if this request is equal to \c MPI_REQUEST_NULL.
    [[nodiscard]] bool is_null() const {
        return _request == MPI_REQUEST_NULL;
    }

    [[nodiscard]] bool test() {
        int is_finished;
        int err = MPI_Test(&_request, &is_finished, MPI_STATUS_IGNORE);
        THROW_IF_MPI_ERROR(err, MPI_Test);
        return is_finished;
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request& native() {
        return _request;
    }

    // TODO: request cancellation and querying of cancelation status

    bool operator==(Request const& other) const {
        return _request == other._request;
    }

    bool operator!=(Request const& other) const {
        return !(*this == other);
    }

private:
    MPI_Request _request; ///< the encapsulated MPI_Request
};

namespace requests {
template <typename... Requests, typename = std::enable_if_t<std::conjunction_v<std::is_same<Requests, Request>...>>>
void wait_all(Requests /*Request*/&... args) {
    constexpr size_t req_size       = sizeof...(args);
    MPI_Request      reqs[req_size] = {args.native()...};
    int              err            = MPI_Waitall(req_size, reqs, MPI_STATUSES_IGNORE);
    THROW_IF_MPI_ERROR(err, MPI_Waitall);
}
// TODO: wait_any, wait_same, test_all, test_any, test_some
} // namespace requests

} // namespace kamping
