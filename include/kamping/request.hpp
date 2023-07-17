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

#include <algorithm>

#include <kamping/error_handling.hpp>
#include <mpi.h>

#include "kamping/data_buffer.hpp"

namespace kamping {

/// @brief Wrapper for MPI request handles (aka. \c MPI_Request).
class Request {
public:
    /// @brief Constructs a request handle from an \c MPI_Request.
    /// @param request The request to encapsulate. Defaults to \c MPI_REQUEST_NULL.
    Request(MPI_Request request = MPI_REQUEST_NULL) : _request(request) {}

    /// @brief Returns when the operation defined by the underlying request completes.
    /// If the underlying request was intialized by a non-blocking communication call, it is set to \c MPI_REQUEST_NULL.
    void wait() {
        int err = MPI_Wait(&_request, MPI_STATUS_IGNORE);
        THROW_IF_MPI_ERROR(err, MPI_Wait);
    }

    /// @return True if this request is equal to \c MPI_REQUEST_NULL.
    [[nodiscard]] bool is_null() const {
        return _request == MPI_REQUEST_NULL;
    }

    /// @return Returns \c true if the underlying request is complete. In that case and if the underlying request was
    /// initialized by a non-blocking communication call, it is set to \c MPI_REQUEST_NULL.
    [[nodiscard]] bool test() {
        int is_finished;
        int err = MPI_Test(&_request, &is_finished, MPI_STATUS_IGNORE);
        THROW_IF_MPI_ERROR(err, MPI_Test);
        return is_finished;
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request& mpi_request() {
        return _request;
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request const& mpi_request() const {
        return _request;
    }

    // TODO: request cancellation and querying of cancelation status

    /// @return Returns \c true if the other request wrapper points to the same request.
    bool operator==(Request const& other) const {
        return _request == other._request;
    }

    /// @return Returns \c true if the other request wrapper points to a different request.
    bool operator!=(Request const& other) const {
        return !(*this == other);
    }

private:
    MPI_Request _request; ///< the encapsulated MPI_Request
};

namespace requests {

/// @brief Waits for completion of all requests handles passed.
/// @param requests A (contiguous) container of \c MPI_Request.
/// @tparam Container The container type.
template <
    typename Container,
    typename std::enable_if<
        internal::has_data_member_v<Container> && std::is_same_v<typename Container::value_type, MPI_Request>,
        bool>::type = true>
void wait_all(Container& requests) {
    int err = MPI_Waitall(asserting_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    THROW_IF_MPI_ERROR(err, MPI_Waitall);
}

/// @brief Waits for completion of all requests handles passed.
/// This incurs overhead for copying the request handles to an intermediate container.
/// @param requests A (contiguous) container of \ref kamping::Request.
/// @tparam Container The container type.
template <
    typename Container,
    typename std::enable_if<
        internal::has_data_member_v<Container> && std::is_same_v<typename Container::value_type, Request>,
        bool>::type = true>
void wait_all(Container const& requests) {
    MPI_Request reqs[requests.size()];
    auto        begin = requests.data();
    auto        end   = begin + requests.size();
    std::transform(begin, end, reqs.begin(), [](Request& req) { return req.mpi_request(); });
    auto req_span = kamping::Span<MPI_Request>(reqs, requests.size());
    wait_all(req_span);
}

/// @brief Wait for completion of all request handles passed.
/// This incurs overhead for copying the request handles to an intermediate container.
/// @param args A list of \ref kamping::Request object to wait on.
template <typename... Requests, typename = std::enable_if_t<std::conjunction_v<std::is_same<Requests, Request>...>>>
void wait_all(Requests /*Request*/&... args) {
    constexpr size_t req_size       = sizeof...(args);
    MPI_Request      reqs[req_size] = {args.mpi_request()...};
    auto             req_span       = kamping::Span<MPI_Request>(reqs, req_size);
    wait_all(req_span);
}

// TODO: wait_any, wait_same, test_all, test_any, test_some

} // namespace requests

} // namespace kamping
