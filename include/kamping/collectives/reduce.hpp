// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <tuple>
#include <type_traits>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping {
namespace internal {
template <typename Communicator>
class Reduce : public CRTPHelper<Communicator, Reduce> {
public:
    template <typename... Args>
    auto reduce(Args&&... args) {
        static_assert(all_parameters_are_rvalues<Args...>);
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
        static_assert(
            std::is_same_v<send_value_type, recv_value_type>, "Types of send and receive buffers do not match.");
        auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
            std::tuple(this->underlying().root()), args...);
        auto&        operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
        auto         operation       = operation_param.template build_operation<send_value_type>();
        MPI_Datatype type            = mpi_datatype<send_value_type>();
        int          err             = MPI_Reduce(
                                 send_buf.ptr, recv_buf.get_ptr(send_buf.size), asserting_cast<int>(send_buf.size), type, operation.op(),
                                 root.rank(), this->underlying().mpi_communicator());
        THROW_IF_MPI_ERROR(err, MPI_Reduce);
        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{});
    }

protected:
    Reduce(){};
};
} // namespace internal
} // namespace kamping
