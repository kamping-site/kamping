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

class RequestPool {
public:
    RequestPool() {}

    using index_type = size_t;

    size_t num_requests() const {
        return _requests.size();
    }

    MPI_Request* request_ptr() {
        return _requests.data();
    }

    inline PooledRequest<index_type> get_request() {
        MPI_Request& req = _requests.emplace_back(MPI_REQUEST_NULL);
        return PooledRequest<index_type>{_requests.size() - 1, req};
    }

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

private:
    std::vector<MPI_Request> _requests;
};
} // namespace kamping
