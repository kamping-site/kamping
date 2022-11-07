// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

/// @brief Wrapper for \c MPI_Send.
///
/// This wraps \c MPI_Send. This operation sends the elements in the input buffer provided via \c
/// kamping::send_buf() to the specified receiver rank using standard send mode.
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent.
/// - \ref kamping::receiver() the receiving rank.
///
/// The following parameters are optional:
/// - \ref kamping::tag() the tag added to the message. Defaults to the communicator's default tag (\ref
/// Communicator::default_tag()) if not present.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
template <typename... Args>
void kamping::Communicator::send(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, receiver), KAMPING_OPTIONAL_PARAMETERS(tag));

    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

    auto const& receiver = internal::select_parameter_type<internal::ParameterType::receiver>(args...);

    using tag_buf_type = decltype(kamping::tag(0));

    int tag = internal::select_parameter_type_or_default<internal::ParameterType::tag, tag_buf_type>(
                  std::tuple(this->default_tag()),
                  args...
    )
                  .get_single_element();
    THROWING_KASSERT(is_valid_tag(tag), "invalid tag " << tag << ", maximum allowed tag is " << tag_upper_bound());

    auto mpi_send_type = mpi_datatype<send_value_type>();

    KASSERT(this->is_valid_rank(receiver.rank()), "Invalid receiver rank.");
    [[maybe_unused]] int err = MPI_Send(
        send_buf.data(),                      // send_buf
        asserting_cast<int>(send_buf.size()), // send_count
        mpi_send_type,                        // send_type
        receiver.rank_signed(),               // receiver
        tag,                                  // tag
        this->mpi_communicator()
    );
    THROW_IF_MPI_ERROR(err, MPI_Send);
}
