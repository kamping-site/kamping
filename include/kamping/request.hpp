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

#include <optional>

#include <kamping/error_handling.hpp>
#include <mpi.h>

#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"

namespace kamping {

/// @brief Wrapper for MPI request handles (aka. \c MPI_Request).
class Request {
public:
    /// @brief Constructs a request handle from an \c MPI_Request.
    /// @param request The request to encapsulate. Defaults to \c MPI_REQUEST_NULL.
    Request(MPI_Request request = MPI_REQUEST_NULL) : _request(request) {}

    /// @brief Returns when the operation defined by the underlying request completes.
    /// If the underlying request was initialized by a non-blocking communication call, it is set to \c
    /// MPI_REQUEST_NULL.
    ///
    /// @param status A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    ///
    /// @return The status object, if \p status is \ref kamping::status_out(), otherwise nothing.
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    auto wait(StatusParamObjectType status = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        int err = MPI_Wait(&_request, internal::status_param_to_native_ptr(status));
        THROW_IF_MPI_ERROR(err, MPI_Wait);
        if constexpr (internal::is_extractable<StatusParamObjectType>) {
            return status.extract();
        }
    }

    /// @return True if this request is equal to \c MPI_REQUEST_NULL.
    [[nodiscard]] bool is_null() const {
        return _request == MPI_REQUEST_NULL;
    }

    /// @brief Tests for completion of the underlying request. If the underlying request was
    /// initialized by a non-blocking communication call and completes, it is set to \c MPI_REQUEST_NULL.
    ///
    /// @param status A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    ///
    /// @return Returns \c true if the underlying request is complete. If \p status is \ref kamping::status_out(),
    /// returns an \c std::optional encapsulating the status in case of completion, \c std::nullopt otherwise.
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    [[nodiscard]] auto test(StatusParamObjectType status = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        int is_finished;
        int err = MPI_Test(&_request, &is_finished, internal::status_param_to_native_ptr(status));
        THROW_IF_MPI_ERROR(err, MPI_Test);
        if constexpr (internal::is_extractable<StatusParamObjectType>) {
            if (is_finished) {
                return std::optional{status.extract()};
            } else {
                return std::optional<Status>{};
            }
        } else {
            return static_cast<bool>(is_finished);
        }
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request& mpi_request() {
        return _request;
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request const& mpi_request() const {
        return _request;
    }

    // TODO: request cancellation and querying of cancellation status

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
        internal::has_data_member_v<
            Container> && std::is_same_v<typename std::remove_reference_t<Container>::value_type, MPI_Request>,
        bool>::type = true>
void wait_all(Container&& requests) {
    int err = MPI_Waitall(asserting_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    THROW_IF_MPI_ERROR(err, MPI_Waitall);
}

/// @brief Waits for completion of all requests handles passed.
/// Warning: This relies on undefined behavior!
/// @param requests A (contiguous) container of \ref kamping::Request.
/// @tparam Container The container type.
template <
    typename Container,
    typename std::enable_if<
        internal::has_data_member_v<
            Container> && std::is_same_v<typename std::remove_reference_t<Container>::value_type, Request>,
        bool>::type = true>
void wait_all_with_undefined_behavior(Container&& requests) {
    static_assert(
        // "A pointer to a standard-layout class may be converted (with reinterpret_cast) to a pointer to its first
        // non-static data member and vice versa." https://en.cppreference.com/w/cpp/types/is_standard_layout
        sizeof(Request) == sizeof(MPI_Request) && std::is_standard_layout_v<Request>,
        "Request is not layout compatible with MPI_Request."
    );
    // this is still undefined, we could cast a single pointer, but not an array
    MPI_Request* begin = reinterpret_cast<MPI_Request*>(requests.data());
    wait_all(Span<MPI_Request>{begin, requests.size()});
}
///
/// @brief Waits for completion of all requests handles passed.
/// This incurs overhead for copying the request handles to an intermediate container.
/// @param requests A (contiguous) container of \ref kamping::Request.
/// @tparam Container The container type.
template <
    typename Container,
    typename std::enable_if<
        internal::has_data_member_v<
            Container> && std::is_same_v<typename std::remove_reference_t<Container>::value_type, Request>,
        bool>::type = true>
void wait_all(Container&& requests) {
    std::vector<MPI_Request> mpi_requests(requests.size());
    // we can not use the STL here, because we only check for the presence of .data()
    auto   begin = requests.data();
    auto   end   = begin + requests.size();
    size_t idx   = 0;
    for (auto current = begin; current != end; current++) {
        mpi_requests[idx] = current->mpi_request();
        idx++;
    }
    wait_all(mpi_requests);
}

/// @brief Wait for completion of all request handles passed.
/// This incurs overhead for copying the request handles to an intermediate container.
/// @param args A list of request handles to wait on. These may be lvalues or rvalues convertible \ref kamping::Request.
template <
    typename... RequestType,
    typename =
        std::enable_if_t<std::conjunction_v<std::is_convertible<std::remove_reference_t<RequestType>, Request>...>>>
void wait_all(RequestType&&... args) {
    constexpr size_t req_size       = sizeof...(args);
    MPI_Request      reqs[req_size] = {Request{args}.mpi_request()...};
    wait_all(Span<MPI_Request>(reqs, req_size));
}

// TODO: wait_any, wait_same, test_all, test_any, test_some

} // namespace requests
} // namespace kamping
