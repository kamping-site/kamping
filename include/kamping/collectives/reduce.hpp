#pragma once

#include "kamping/checking_casts.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"
#include "kamping/error_handling.hpp"

#include <mpi.h>
#include <tuple>
#include <type_traits>

namespace kamping {
namespace internal {
template <typename Communicator, typename... Args>
auto reduce(const Communicator& comm, Args&&... args) {
    static_assert(
        internal::has_parameter_type<internal::ParameterType::send_buf, Args...>(),
        "Missing required parameter send_buf.");
    static_assert(
        internal::has_parameter_type<internal::ParameterType::op, Args...>(), "Missing required parameter op.");

    auto& send_buf_param        = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf              = send_buf_param.get();
    using send_value_type       = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(std::is_same_v<send_value_type, recv_value_type>, "Types of send and receive buffers do not match.");
    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
        std::tuple(comm.root()), args...);
    auto&        operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
    auto         operation       = operation_param.template build_operation<send_value_type>();
    MPI_Datatype type            = mpi_datatype<send_value_type>();
    int          err             = MPI_Reduce(
                             send_buf.ptr, recv_buf.get_ptr(send_buf.size), asserting_cast<int>(send_buf.size), type, operation.op(),
                             root.rank(), comm.mpi_communicator());
    // TODO throw correct Exception with propagated error code
    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return MPIResult(
        std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
        internal::BufferCategoryNotUsed{});
}
} // namespace internal
} // namespace kamping
