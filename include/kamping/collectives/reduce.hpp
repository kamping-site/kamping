#pragma once

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"
#include <mpi.h>

namespace kamping {
class Reduce {
public:
    template <typename... Args>
    auto reduce(Communicator& comm, Args&&... args) {
        auto send_buf         = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
        using send_value_type = typename decltype(send_buf)::value_type;
        auto default_recv_buf = kamping::recv_buf(NewContainer<std::vector<send_value_type>>{});
        auto recv_buf = internal::select_parameter_type<internal::ParameterType::recv_buf>(args..., default_recv_buf);
        using recv_value_type = typename decltype(recv_buf)::value_type;
        static_assert(
            std::is_same_v<send_value_type, recv_value_type>, "Types of send and receive buffers do not match.");
        auto default_root = kamping::root(comm.root());
        auto root         = internal::select_parameter_type<internal::ParameterType::root>(args..., default_root);
        auto operation    = internal::select_parameter_type<internal::ParameterType::op>(args...)
                             .template build_operation<send_value_type>();
        MPI_Datatype type = mpi_datatype<send_value_type>();
        MPI_Reduce(
            send_buf.ptr, recv_buf.get_ptr(send_buf.size), asserting_cast<int>(send_buf.size), type, operation.op(),
            root.rank(), comm.mpi_communicator());
        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{});
    }
};
} // namespace kamping
