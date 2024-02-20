// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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
#pragma once

#include <cstddef>
#include <vector>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/request.hpp"

namespace kamping {

/// @brief Result returned by \ref RequestPool.wait_any()
/// @tparam IndexType Type of the stored Index.
/// @tparam StatusType Type of the status object.
template <typename IndexType, typename StatusType>
struct PoolAnyResult {
    IndexType
        index; ///< The index of the completed operation. \ref RequestPool.index_end() if there were no active requests.
    StatusType status; ///< The status of the complete operation.
};

/// @brief A pool for storing multiple \ref Request s and checking them for completion.
///
/// Requests are internally stored in a vector. The vector is resized as needed.
/// New requests can be obtained by calling \ref get_request.
///
/// @tparam DefaultContainerType The default container type to use for containers created inside pool operations.
/// Defaults to std::vector.
template <template <typename...> typename DefaultContainerType = std::vector>
class RequestPool {
public:
    /// @brief Constructs a new empty \ref RequestPool.
    RequestPool() {}

    using index_type = size_t; ///< The type used to index requests in the pool.

    /// @brief Type of the default container type to use for containers created inside operations of this request pool.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    /// @brief The first index value. The pool is empty if `index_begin() == index_end()`.
    index_type index_begin() const {
        return 0;
    }

    /// @brief The index value after the last one. The pool is empty if `index_begin() == index_end()`.
    index_type index_end() const {
        return _requests.size();
    }

    /// @brief Returns the number of requests currently stored in the pool.
    size_t num_requests() const {
        return _requests.size();
    }

    /// @brief Returns a pointer to the underlying MPI_Request array.
    MPI_Request* request_ptr() {
        return _requests.data();
    }

    /// @brief Adds a new request to the pool and returns a \ref PooledRequest encapsulating it.
    inline PooledRequest<index_type> get_request() {
        MPI_Request& req = _requests.emplace_back(MPI_REQUEST_NULL);
        return PooledRequest<index_type>{_requests.size() - 1, req};
    }

    /// @brief Waits for all requests in the pool to complete by calling \c MPI_Waitall.
    /// @param statuses_param A \c statuses parameter object to which the status information is written. Defaults
    /// to \c kamping::statuses(ignore<>).
    /// @return If \p statuses is an owning out parameter, returns the status information, otherwise returns nothing.
    template <typename StatusesParamObjectType = decltype(kamping::statuses(ignore<>))>
    auto wait_all(StatusesParamObjectType statuses_param = kamping::statuses(ignore<>)) {
        static_assert(
            StatusesParamObjectType::parameter_type == internal::ParameterType::statuses,
            "Only statuses parameters are allowed."
        );
        auto        statuses = statuses_param.template construct_buffer_or_rebind<DefaultContainerType>();
        MPI_Status* statuses_ptr;
        if constexpr (decltype(statuses)::buffer_type == internal::BufferType::ignore) {
            statuses_ptr = MPI_STATUS_IGNORE;
        } else {
            auto compute_requested_size = [&] {
                return num_requests();
            };
            statuses.resize_if_requested(compute_requested_size);
            KASSERT(
                statuses.size() >= compute_requested_size(),
                "statuses buffer is not large enough to hold all status information.",
                assert::light
            );
            statuses_ptr = statuses.data();
        }
        [[maybe_unused]] int err = MPI_Waitall(asserting_cast<int>(num_requests()), request_ptr(), statuses_ptr);
        THROW_IF_MPI_ERROR(err, MPI_Waitall);
        if constexpr (internal::is_extractable<decltype(statuses)>) {
            return statuses.extract();
        }
    }

    /// @brief Tests whether all requests in the pool have completed by calling \c MPI_Testall.
    /// @param statuses_param A \c statuses parameter object to which the status information is written. Defaults
    /// to \c kamping::statuses(ignore<>).
    /// @return A truthful value if all requests have completed, a falsy value otherwise.
    /// @note By default, returns a \c bool indicated completion, but if \p statuses is an owning out parameter, returns
    /// a \c std::optional containing the status information.
    /// @warning If the status parameter is provided, the underlying buffer is always resized to fit all requests
    /// according to its \c resize_policy, even if not all requests have completed yet. This is because MPI
    /// does not allow retrieving statuses after a test succeeded.
    template <typename StatusesParamObjectType = decltype(kamping::statuses(ignore<>))>
    auto test_all(StatusesParamObjectType statuses_param = kamping::statuses(ignore<>)) {
        static_assert(
            StatusesParamObjectType::parameter_type == internal::ParameterType::statuses,
            "Only statuses parameters are allowed."
        );
        auto        statuses = statuses_param.template construct_buffer_or_rebind<DefaultContainerType>();
        MPI_Status* statuses_ptr;
        if constexpr (decltype(statuses)::buffer_type == internal::BufferType::ignore) {
            statuses_ptr = MPI_STATUS_IGNORE;
        } else {
            auto compute_requested_size = [&] {
                return num_requests();
            };
            statuses.resize_if_requested(compute_requested_size);
            KASSERT(
                statuses.size() >= compute_requested_size(),
                "statuses buffer is not large enough to hold all status information.",
                assert::light
            );
            statuses_ptr = statuses.data();
        }
        int                  succeeded = false;
        [[maybe_unused]] int err =
            MPI_Testall(asserting_cast<int>(num_requests()), request_ptr(), &succeeded, statuses_ptr);
        THROW_IF_MPI_ERROR(err, MPI_Testall);
        if constexpr (internal::is_extractable<decltype(statuses)>) {
            if (succeeded) {
                return std::optional{statuses.extract()};
            } else {
                return std::optional<decltype(statuses.extract())>{};
            }
        } else {
            return static_cast<bool>(succeeded);
        }
    }

    /// @brief Waits any request in the pool to complete by calling \c MPI_Waitany.
    /// @param status_param A \c status parameter object to which the status information about the completed operation
    /// is written. Defaults to \c kamping::status(ignore<>).
    /// @return By default, returns  the index of the completed operation. If the pool is
    /// empty or no request in the pool is active, returns an index equal to `index_end()`. If \p status is an owning
    /// out parameter, also returns the status alongside the index by returning a \ref PoolAnyResult.
    /// @see PoolAnyResult
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    auto wait_any(StatusParamObjectType status_param = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        auto status = status_param.construct_buffer_or_rebind();
        int  index;
        int  err = MPI_Waitany(
            asserting_cast<int>(num_requests()),
            request_ptr(),
            &index,
            internal::status_param_to_native_ptr(status)
        );
        THROW_IF_MPI_ERROR(err, MPI_Waitany);
        if constexpr (internal::is_extractable<decltype(status)>) {
            using status_type = decltype(status.extract());
            if (index == MPI_UNDEFINED) {
                return PoolAnyResult<index_type, status_type>{index_end(), status.extract()};
            } else {
                return PoolAnyResult<index_type, status_type>{static_cast<index_type>(index), status.extract()};
            }
        } else {
            if (index == MPI_UNDEFINED) {
                return index_end();
            } else {
                return static_cast<index_type>(index);
            }
        }
    }

    /// @brief Tests if any request in the pool is completed by calling \c MPI_Testany.
    /// @param status_param A \c status parameter object to which the status information about the completed operation
    /// is written. Defaults to \c kamping::status(ignore<>).
    /// @return If any request completes, returns an `std::optional` containing information about the completed request.
    /// Otherwise, `std::nullopt`. The value contained inside the optional depends on \p status parameter and
    /// follows the same rules as for \ref wait_any.
    /// @see wait_any
    template <typename StatusParamObjectType = decltype(status(ignore<>))>
    auto test_any(StatusParamObjectType status_param = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        auto status = status_param.construct_buffer_or_rebind();
        int  index;
        int  flag;
        int  err = MPI_Testany(
            asserting_cast<int>(num_requests()),
            request_ptr(),
            &index,
            &flag,
            internal::status_param_to_native_ptr(status)
        );
        THROW_IF_MPI_ERROR(err, MPI_Testany);
        if constexpr (internal::is_extractable<decltype(status)>) {
            using status_type = decltype(status.extract());
            using return_type = PoolAnyResult<index_type, status_type>;
            if (flag) {
                if (index == MPI_UNDEFINED) {
                    return std::optional<return_type>{{index_end(), status.extract()}};
                } else {
                    return std::optional<return_type>{{static_cast<index_type>(index), status.extract()}};
                }
            } else {
                return std::optional<PoolAnyResult<index_type, status_type>>{};
            }
        } else {
            if (flag) {
                if (index == MPI_UNDEFINED) {
                    return std::optional{index_end()};
                } else {
                    return std::optional{static_cast<index_type>(index)};
                }
            } else {
                return std::optional<index_type>{};
            }
        }
    }

private:
    std::vector<MPI_Request> _requests;
};
} // namespace kamping
