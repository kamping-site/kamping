#include <cstddef>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/ibarrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameter_filtering.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/p2p/iprobe.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/plugin/plugin_helpers.hpp"
#include "kamping/request_pool.hpp"
#include "kamping/result.hpp"

/// @file
/// @brief File containing the SparseAlltoall plugin.

#pragma once
namespace kamping::plugin {
/// @brief Plugin providing a sparse alltoall exchange method.
/// @see \ref ChunkedAlltoall::alltoallv_chunked() for more information.
template <typename Comm, template <typename...> typename DefaultContainerType>
class ChunkedAlltoall : public plugin::PluginBase<Comm, DefaultContainerType, ChunkedAlltoall> {
public:
    template <typename... Args>
    auto alltoallv_chunked(std::size_t k, Args... args) const;
};

template <typename Comm, template <typename...> typename DefaultContainerType>
template <typename... Args>
auto ChunkedAlltoall<Comm, DefaultContainerType>::alltoallv_chunked(std::size_t k, Args... args) const {
    auto& self = this->to_communicator();
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
        KAMPING_OPTIONAL_PARAMETERS(recv_counts, recv_buf, send_displs, recv_displs, send_type, recv_type)
    );

    // Get send_buf
    auto const& send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    // Get recv_buf
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    // Get send/recv types
    auto [send_type, recv_type] =
        internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_type_has_to_be_deduced = internal::has_to_be_computed<decltype(send_type)>;
    [[maybe_unused]] constexpr bool recv_type_has_to_be_deduced = internal::has_to_be_computed<decltype(recv_type)>;

    // Get send_counts
    auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                  .template construct_buffer_or_rebind<DefaultContainerType>();
    using send_counts_type = typename std::remove_reference_t<decltype(send_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_counts_type>, int>, "Send counts must be of type int");
    static_assert(
        !internal::has_to_be_computed<decltype(send_counts)>,
        "Send counts must be given as an input parameter"
    );
    KASSERT(send_counts.size() >= self.size(), "Send counts buffer is not large enough.", assert::light);

    // Get recv_counts
    using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
    auto recv_counts =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_counts_type = typename std::remove_reference_t<decltype(recv_counts)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_counts_type>, int>, "Recv counts must be of type int");

    // Get send_displs
    using default_send_displs_type = decltype(kamping::send_displs_out(alloc_new<DefaultContainerType<int>>));
    auto send_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using send_displs_type = typename std::remove_reference_t<decltype(send_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<send_displs_type>, int>, "Send displs must be of type int");

    // Get recv_displs
    using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
    auto recv_displs =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
    static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

    // Calculate recv_counts if necessary
    constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;

    if constexpr (do_calculate_recv_counts) {
        /// @todo make it possible to test whether this additional communication is skipped
        recv_counts.resize_if_requested([&]() { return self.size(); });
        KASSERT(recv_counts.size() >= self.size(), "Recv counts buffer is not large enough.", assert::light);
        self.alltoall(kamping::send_buf(send_counts.get()), kamping::recv_buf(recv_counts.get()));
    } else {
        KASSERT(recv_counts.size() >= self.size(), "Recv counts buffer is not large enough.", assert::light);
    }

    // Calculate send_displs if necessary
    constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;

    if constexpr (do_calculate_send_displs) {
        send_displs.resize_if_requested([&]() { return self.size(); });
        KASSERT(send_displs.size() >= self.size(), "Send displs buffer is not large enough.", assert::light);
        std::exclusive_scan(send_counts.data(), send_counts.data() + self.size(), send_displs.data(), 0);
    } else {
        KASSERT(send_displs.size() >= self.size(), "Send displs buffer is not large enough.", assert::light);
    }

    // Check that send displs and send counts are large enough
    KASSERT(
        // if the send type is user provided, kamping cannot make any assumptions about the size of the send
        // buffer
        !send_type_has_to_be_deduced
            || *(send_counts.data() + self.size() - 1) +       // Last element of send_counts
                       *(send_displs.data() + self.size() - 1) // Last element of send_displs
                   <= asserting_cast<int>(send_buf.size()),
        assert::light
    );

    // Calculate recv_displs if necessary
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;
    if constexpr (do_calculate_recv_displs) {
        recv_displs.resize_if_requested([&]() { return self.size(); });
        KASSERT(recv_displs.size() >= self.size(), "Recv displs buffer is not large enough.", assert::light);
        std::exclusive_scan(recv_counts.data(), recv_counts.data() + self.size(), recv_displs.data(), 0);
    } else {
        KASSERT(recv_displs.size() >= self.size(), "Recv displs buffer is not large enough.", assert::light);
    }

    auto compute_required_recv_buf_size = [&]() {
        return compute_required_recv_buf_size_in_vectorized_communication(recv_counts, recv_displs, self.size());
    };

    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        // if the recv type is user provided, kamping cannot make any assumptions about the required size of the recv
        // buffer
        !recv_type_has_to_be_deduced || recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    {
        KASSERT(k > 0);
        std::vector<int> chunked_send_counts(self.size());
        std::vector<int> chunked_send_displs(self.size());
        std::vector<int> chunked_recv_counts(self.size());
        std::vector<int> chunked_recv_displs(self.size());
        std::copy_n(send_displs.data(), send_displs.size(), chunked_send_displs.data());
        std::copy_n(recv_displs.data(), recv_displs.size(), chunked_recv_displs.data());
        for (std::size_t i = 1; i < k; ++i) {
            for (std::size_t j = 0; j < self.size(); ++j) {
                chunked_send_counts[j] = send_counts.data()[j] / static_cast<int>(k);
                chunked_recv_counts[j] = recv_counts.data()[j] / static_cast<int>(k);
            }
            // Do the actual alltoallv
            [[maybe_unused]] int err = MPI_Alltoallv(
                send_buf.data(),                // send_buf
                chunked_send_counts.data(),     // send_counts
                chunked_send_displs.data(),     // send_displs
                send_type.get_single_element(), // send_type
                recv_buf.data(),                // send_counts
                chunked_recv_counts.data(),     // recv_counts
                chunked_recv_displs.data(),     // recv_displs
                recv_type.get_single_element(), // recv_type
                self.mpi_communicator()         // comm
            );
            self.mpi_error_hook(err, "MPI_Alltoallv");
            for (std::size_t j = 0; j < self.size(); ++j) {
                chunked_send_displs[j] += chunked_send_counts[j];
                chunked_recv_displs[j] += chunked_recv_counts[j];
            }
        }
        for (std::size_t j = 0; j < self.size(); ++j) {
            chunked_send_counts[j] = send_counts.data()[j] - chunked_send_counts[j];
            chunked_recv_counts[j] = recv_counts.data()[j] - chunked_recv_counts[j];
        }
        // Do the actual alltoallv
        [[maybe_unused]] int err = MPI_Alltoallv(
            send_buf.data(),                // send_buf
            chunked_send_counts.data(),     // send_counts
            chunked_send_displs.data(),     // send_displs
            send_type.get_single_element(), // send_type
            recv_buf.data(),                // send_counts
            chunked_recv_counts.data(),     // recv_counts
            chunked_recv_displs.data(),     // recv_displs
            recv_type.get_single_element(), // recv_type
            self.mpi_communicator()         // comm
        );
        self.mpi_error_hook(err, "MPI_Alltoallv");
    }

    return internal::make_mpi_result<std::tuple<Args...>>(
        std::move(recv_buf),    // recv_buf
        std::move(recv_counts), // recv_counts
        std::move(recv_displs), // recv_displs
        std::move(send_displs), // send_displs
        std::move(send_type),   // send_type
        std::move(recv_type)    // recv_type
    );
}

} // namespace kamping::plugin
