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

/// @brief A pool for storing multiple \ref Request s and checking them for completion.
///
/// Requests are internally stored in a vector. The vector is resized as needed.
/// New requests can be obtained by calling \ref get_request.
class RequestPool {
public:
    /// @brief Constructs a new empty \ref RequestPool.
    RequestPool() {}

    using index_type = size_t; ///< The type used to index requests in the pool.

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
    /// @param statuses A \ref statuses parameter object to which the status information is written. Defaults to \c
    /// kamping::statuses(ignore<>).
    /// @return If \p statuses is an owning out parameter, returns the status information, otherwise returns nothing.
    template <typename StatusesParamObjectType = decltype(kamping::statuses(ignore<>))>
    auto wait_all(StatusesParamObjectType statuses = kamping::statuses(ignore<>)) {
        static_assert(
            StatusesParamObjectType::parameter_type == internal::ParameterType::statuses,
            "Only statuses parameters are allowed."
        );
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
    /// @param statuses A \ref kamping::statuses parameter object to which the status information is written. Defaults
    /// to \c kamping::statuses(ignore<>).
    /// @return A thruthful value if all requests have completed, a falsy value otherwise.
    /// @note By default, returns a \c bool indicated completion, but if \p statuses is an owning out parameter, returns
    /// a \c std::optional containing the status information.
    /// @warning If the status parameter is provided, the underlying buffer is always resized to fit all requests
    /// according to its \ref kamping::BufferResizePolicy, even not all requests have completed yet. This is because MPI
    /// does not allow retrieving statuses after a test succeeded.
    template <typename StatusesParamObjectType = decltype(kamping::statuses(ignore<>))>
    auto test_all(StatusesParamObjectType statuses = kamping::statuses(ignore<>)) {
        static_assert(
            StatusesParamObjectType::parameter_type == internal::ParameterType::statuses,
            "Only statuses parameters are allowed."
        );
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

private:
    std::vector<MPI_Request> _requests;
};
} // namespace kamping
