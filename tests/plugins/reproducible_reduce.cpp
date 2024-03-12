#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

// Binary Tree Reduce

template <template <typename...> typename DefaultContainerType>
class ReproducibleCommunicator {
public:
    using Communicator = kamping::Communicator<DefaultContainerType>;

    template <
        template <typename...> typename = DefaultContainerType,
        template <typename, template <typename...> typename>
        typename... Plugins>
    ReproducibleCommunicator(
        kamping::Communicator<DefaultContainerType, Plugins...> const& comm, std::map<size_t, size_t> start_indices
    )
        : _start_indices{start_indices} {
        // TODO: how to simply set communicator without this identity split operation (copied from alltoall plugin)
        auto mycomm = comm.split(0);
        _comm       = Communicator(mycomm.disown_mpi_communicator(), mycomm.root_signed(), true);
        KASSERT(_comm.size() == comm.size());
    }

    template <typename... Args>
    void reproducible_reduce(Args... args) {
        using namespace kamping;
        KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS());

        // TODO: consider root parameter

        // get send buffer
        auto&& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        // using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    }

private:
    kamping::Communicator<DefaultContainerType> _comm;
    const std::map<size_t, size_t>              _start_indices;
};

// Plugin Code
template <typename Comm, template <typename...> typename DefaultContainerType>
class ReproducibleReducePlugin
    : public kamping::plugin::PluginBase<Comm, DefaultContainerType, ReproducibleReducePlugin> {
public:
    template <typename... Args>
    auto make_reproducible_comm(Args... args) {
        using namespace kamping;

        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(recv_displs, send_counts),
            KAMPING_OPTIONAL_PARAMETERS()
        );

        using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
        auto&& recv_displs =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
        static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

        auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();
        // This is the value type (i.e. of the underlying container)
        // using sendcounts_type = typename std::remove_reference_t<decltype(send_counts)>::value_type;

        auto comm = this->to_communicator();
        KASSERT(send_counts.size() == comm.size(), "send_counts must be of same size as communicator");
        KASSERT(recv_displs.size() == comm.size(), "recv_displs must be of same size as communicator");

        auto const global_array_length = static_cast<size_t>(
            std::reduce(send_counts.data(), send_counts.data() + send_counts.size(), 0, std::plus<>())
        );

        // Construct index map which maps global array indices to PEs
        std::map<size_t, size_t> start_indices;
        for (size_t p = 0; p < comm.size(); ++p) {
            KASSERT(send_counts.data()[p] >= 0, "send_count for rank " << p << " is negative");
            KASSERT(recv_displs.data()[p] >= 0, "displacement for rank " << p << " is negative");

            if (send_counts.data()[p] == 0)
                continue;

            start_indices[asserting_cast<size_t>(recv_displs.data()[p])] = p;
        }
        start_indices[global_array_length] = comm.size(); // guardian element

        KASSERT(start_indices.find(0) != start_indices.end(), "recv_displs does not have entry for index 0");
        // Verify correctness of index map
        for (auto it = start_indices.begin(); it != start_indices.end(); ++it) {
            auto const next = std::next(it);
            if (next == start_indices.end())
                break;

            auto const rank              = it->second;
            auto const region_start      = it->first;
            auto const region_end        = region_start + asserting_cast<size_t>(send_counts.data()[rank]);
            auto const next_rank         = next->second;
            auto const next_region_start = next->first;

            KASSERT(
                region_end == next_region_start,
                "Region of rank " << rank << " ends at index " << region_end << ", but next region of rank "
                                  << next_rank << " starts at index " << next_region_start
            );
        }

        return ReproducibleCommunicator<DefaultContainerType>(this->to_communicator(), start_indices);
    }
};

TEST(ReproducibleReduceTest, PluginInit) {
    kamping::Communicator<std::vector, ReproducibleReducePlugin> comm;

    double const        epsilon = std::numeric_limits<double>::epsilon();
    std::vector<double> test_array{1, 1 + epsilon, 2 + epsilon, epsilon, 8, 9};

    int              values_per_rank = test_array.size() / comm.size();
    std::vector<int> send_counts(comm.size(), values_per_rank);
    std::vector<int> recv_displs;

    for (size_t i = 0; i < comm.size(); i++) {
        recv_displs.push_back(i);
        i += kamping::asserting_cast<size_t>(send_counts[i]);
    }

    auto reproducible_comm =
        comm.make_reproducible_comm(kamping::recv_displs(recv_displs), kamping::send_counts(send_counts));

    // comm.reproducible_reduce();

    if (comm.is_root()) {
    } else if (comm.rank() == 1) {
    }
}
