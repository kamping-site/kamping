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
/// - \ref kamping::send_mode() the send mode to use. Defaults to standard MPI_Send.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
template <typename... Args>
void kamping::Communicator::send(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, receiver),
        KAMPING_OPTIONAL_PARAMETERS(tag, send_mode)
    );

    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

    auto const& receiver = internal::select_parameter_type<internal::ParameterType::receiver>(args...);

    using default_tag_buf_type = decltype(kamping::tag(0));

    int tag = internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>(
                  std::tuple(this->default_tag()),
                  args...
    )
                  .get_single_element();
    KASSERT(is_valid_tag(tag), "invalid tag " << tag << ", maximum allowed tag is " << tag_upper_bound());

    using send_mode_obj_type = decltype(internal::select_parameter_type_or_default<
                                        internal::ParameterType::send_mode,
                                        internal::SendModeParameter<internal::standard_mode_t>>(std::tuple(), args...));
    using send_mode          = typename std::remove_reference_t<send_mode_obj_type>::send_mode;

    auto mpi_send_type = mpi_datatype<send_value_type>();

    KASSERT(this->is_valid_rank(receiver.rank()), "Invalid receiver rank.");

    if constexpr (std::is_same_v<send_mode, internal::standard_mode_t>) {
        [[maybe_unused]] int err = MPI_Send(
            send_buf.data(),                      // send_buf
            asserting_cast<int>(send_buf.size()), // send_count
            mpi_send_type,                        // send_type
            receiver.rank_signed(),               // receiver
            tag,                                  // tag
            this->mpi_communicator()
        );
        THROW_IF_MPI_ERROR(err, MPI_Send);
    } else if constexpr (std::is_same_v<send_mode, internal::buffered_mode_t>) {
        [[maybe_unused]] int err = MPI_Bsend(
            send_buf.data(),                      // send_buf
            asserting_cast<int>(send_buf.size()), // send_count
            mpi_send_type,                        // send_type
            receiver.rank_signed(),               // receiver
            tag,                                  // tag
            this->mpi_communicator()
        );
        THROW_IF_MPI_ERROR(err, MPI_Bsend);
    } else if constexpr (std::is_same_v<send_mode, internal::synchronous_mode_t>) {
        [[maybe_unused]] int err = MPI_Ssend(
            send_buf.data(),                      // send_buf
            asserting_cast<int>(send_buf.size()), // send_count
            mpi_send_type,                        // send_type
            receiver.rank_signed(),               // receiver
            tag,                                  // tag
            this->mpi_communicator()
        );
        THROW_IF_MPI_ERROR(err, MPI_Ssend);
    } else if constexpr (std::is_same_v<send_mode, internal::ready_mode_t>) {
        [[maybe_unused]] int err = MPI_Rsend(
            send_buf.data(),                      // send_buf
            asserting_cast<int>(send_buf.size()), // send_count
            mpi_send_type,                        // send_type
            receiver.rank_signed(),               // receiver
            tag,                                  // tag
            this->mpi_communicator()
        );
        THROW_IF_MPI_ERROR(err, MPI_Rsend);
    }
}

/// @brief Convenience wrapper for MPI_Bsend. Calls \ref kamping::Communicator::send() with the appropriate send mode
/// set.
template <typename... Args>
void kamping::Communicator::bsend(Args... args) const {
    this->send(std::forward<Args>(args)..., send_mode(send_modes::buffered));
}

/// @brief Convenience wrapper for MPI_Ssend. Calls \ref kamping::Communicator::send() with the appropriate send mode
/// set.
template <typename... Args>
void kamping::Communicator::ssend(Args... args) const {
    this->send(std::forward<Args>(args)..., send_mode(send_modes::synchronous));
}

/// @brief Convenience wrapper for MPI_Rsend. Calls \ref kamping::Communicator::send() with the appropriate send mode
/// set.
template <typename... Args>
void kamping::Communicator::rsend(Args... args) const {
    this->send(std::forward<Args>(args)..., send_mode(send_modes::ready));
}