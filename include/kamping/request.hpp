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

#include <mpi.h>

#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"

namespace kamping {

/// @brief Base class for request wrappers.
///
/// This class provides the common interface for all request wrappers. It is
/// not intended to be used directly. Instead, use \ref kamping::Request or
/// \ref kamping::PooledRequest or define your own request type, which must
/// implement \c request_ptr().
///
/// @tparam RequestType The derived type.
template <typename RequestType>
class RequestBase {
public:
    constexpr RequestBase() = default;
    ~RequestBase()          = default;

    /// @brief Copy constructor is deleted because requests should only be moved.
    RequestBase(RequestBase const&) = delete;
    /// @brief Copy assignment operator is deleted because requests should only be moved.
    RequestBase& operator=(RequestBase const&) = delete;
    /// @brief Move constructor.
    RequestBase(RequestBase&&) = default;
    /// @brief Move assignment operator.
    RequestBase& operator=(RequestBase&&) = default;

private:
    ///@brief returns a pointer to the wrapped MPI_Request by calling \c request_ptr() on \ref RequestType using CRTP.
    MPI_Request* request_ptr() {
        return static_cast<RequestType&>(*this).request_ptr();
    }
    ///@brief returns a const pointer to the wrapped MPI_Request by calling \c request_ptr() on \ref RequestType using
    /// CRTP.
    MPI_Request const* request_ptr() const {
        return static_cast<RequestType const&>(*this).request_ptr();
    }

public:
    /// @brief Returns when the operation defined by the underlying request completes.
    /// If the underlying request was initialized by a non-blocking communication call, it is set to \c
    /// MPI_REQUEST_NULL.
    ///
    /// @param status_param A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    ///
    /// @return The status object, if \p status is \ref kamping::status_out(), otherwise nothing.
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    auto wait(StatusParamObjectType status_param = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        auto status = status_param.construct_buffer_or_rebind();
        int  err    = MPI_Wait(request_ptr(), internal::status_param_to_native_ptr(status));
        THROW_IF_MPI_ERROR(err, MPI_Wait);
        if constexpr (internal::is_extractable<StatusParamObjectType>) {
            return status.extract();
        }
    }

    /// @return True if this request is equal to \c MPI_REQUEST_NULL.
    [[nodiscard]] bool is_null() const {
        return *request_ptr() == MPI_REQUEST_NULL;
    }

    /// @brief Tests for completion of the underlying request. If the underlying request was
    /// initialized by a non-blocking communication call and completes, it is set to \c MPI_REQUEST_NULL.
    ///
    /// @param status_param A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    ///
    /// @return Returns \c true if the underlying request is complete. If \p status is \ref kamping::status_out() and
    /// owning, returns an \c std::optional encapsulating the status in case of completion, \c std::nullopt otherwise.
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    [[nodiscard]] auto test(StatusParamObjectType status_param = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        auto status = status_param.construct_buffer_or_rebind();
        int  is_finished;
        int  err = MPI_Test(request_ptr(), &is_finished, internal::status_param_to_native_ptr(status));
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
        return *request_ptr();
    }

    /// @return A reference to the underlying MPI_Request handle.
    [[nodiscard]] MPI_Request const& mpi_request() const {
        return *request_ptr();
    }

    // TODO: request cancellation and querying of cancellation status

    /// @return Returns \c true if the other request wrapper points to the same request.
    template <typename T>
    bool operator==(RequestBase<T> const& other) const {
        return *request_ptr() == *other.request_ptr();
    }

    /// @return Returns \c true if the other request wrapper points to a different request.
    template <typename T>
    bool operator!=(RequestBase<T> const& other) const {
        return !(*this == other);
    }
};

/// @brief Wrapper for MPI request handles (aka. \c MPI_Request).
class Request : public RequestBase<Request> {
public:
    /// @brief Constructs a request handle from an \c MPI_Request.
    /// @param request The request to encapsulate. Defaults to \c MPI_REQUEST_NULL.
    Request(MPI_Request request = MPI_REQUEST_NULL) : _request(request) {}

    /// @brief returns a pointer to the wrapped MPI_Request.
    MPI_Request* request_ptr() {
        return &_request;
    }

    /// @brief returns a const pointer to the wrapped MPI_Request.
    MPI_Request const* request_ptr() const {
        return &_request;
    }

private:
    MPI_Request _request; ///< the encapsulated MPI_Request
};

/// @brief Wrapper for MPI requests owned by a \ref RequestPool.
///
/// @tparam IndexType type of the index of this request in the pool.
template <typename IndexType>
class PooledRequest : public RequestBase<PooledRequest<IndexType>> {
public:
    /// @brief constructs a \ref PooledRequest with the given index \p idx and \p request.
    PooledRequest(IndexType idx, MPI_Request& request) : _index(idx), _request(request) {}

    /// @brief returns a pointer to the wrapped MPI_Request.
    MPI_Request* request_ptr() {
        return &_request;
    }

    /// @brief returns a const pointer to the wrapped MPI_Request.
    MPI_Request const* request_ptr() const {
        return &_request;
    }

    /// @brief provides access to this request's index in the pool.
    IndexType index() const {
        return _index;
    }

private:
    IndexType    _index;   ///< the index
    MPI_Request& _request; ///< the encapsulated request
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
    MPI_Request      reqs[req_size] = {Request{std::move(args)}.mpi_request()...};
    wait_all(Span<MPI_Request>(reqs, req_size));
}

// TODO: wait_any, wait_same, test_all, test_any, test_some

} // namespace requests
} // namespace kamping
