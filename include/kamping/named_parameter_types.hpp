// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief File containing the parameter types used by the KaMPIng library

#pragma once

namespace kamping {
/// @brief Internal namespace marking the code that is not user-facing.
///
namespace internal {

/// @addtogroup kamping_utility
/// @{

/// @brief Each input parameter to one of the \c MPI calls wrapped by KaMPIng needs to has one of the following tags.
///
/// The \c MPI calls wrapped by KaMPIng do not rely on the restricting positional parameter paradigm but use named
/// parameters instead. The ParameterTypes defined in this enum are necessary to implement this approach, as KaMPIng
/// needs to identify the purpose of each (unordered) argument.
/// Note that not all enum entries are necessary in each wrapped \c MPI call.
enum class ParameterType {
    send_buf,         ///< Tag used to represent a send buffer, i.e. a buffer containing
                      ///< the data elements to be sent via \c MPI.
    recv_buf,         ///< Tag used to represent a receive buffer, i.e. a buffer
                      ///< containing the data elements to be received via \c MPI.
    send_recv_buf,    ///< Tag used to represent a send and receive buffer, i.e. a
                      ///< buffer containing the data elements to be sent or received
                      ///< (depending on the process' rank) via \c MPI.
    recv_counts,      ///< Tag used to represent a receive counts buffer, i.e. a buffer
                      ///< containing the receive counts from the involved PEs.
    recv_count,       ///< Tag used to represent the number of elements to be received.
    recv_displs,      ///< Tag used to represent a receive displacements buffer, i.e. a
                      ///< buffer containing the receive displacements from the
                      ///< involved PEs.
    send_counts,      ///< Tag used to represent a send counts buffer, i.e. a buffer
                      ///< containing the send counts from the involved PEs.
    send_count,       ///< Tag used to represent the number of elements to be sent.
    send_displs,      ///< Tag used to represent a send displacements buffer, i.e. a
                      ///< buffer containing the send displacements from the involved
                      ///< PEs.
    send_recv_count,  ///< Tag used to represent the number of elements to be sent or
                      ///< received.
    op,               ///< Tag used to represent a reduce operation in a \c MPI call.
    source,           ///< Tag used to represent the sending PE in a \c MPI call.
    destination,      ///< Tag used to represent the receiving PE in a \c MPI call.
    status,           ///< Tag used to represent the status in a \c MPI call.
    statuses,         ///< Tag used to represent a container of statuses in a \c MPI call.
    request,          ///< Tag used to represent an \c MPI_Request.
    root,             ///< Tag used to represent the root PE in a \c MPI collectives call.
    tag,              ///< Tag used to represent the message tag in a \c MPI call.
    send_tag,         ///< Tag used to represent the message send tag in a \c MPI call.
    recv_tag,         ///< Tag used to represent the message recv tag in a \c MPI call.
    send_mode,        ///< Tag used to represent the send mode used by a send operation.
    values_on_rank_0, ///< Tag used to represent the value of the exclusive scan
                      ///< operation on rank 0.
    send_type,        ///< Tag used to represent a send type in an \c MPI call.
    recv_type,        ///< Tag used to represent a recv type in an \c MPI call.
    send_recv_type,   ///< Tag used to represent a send and/or recv type in an \c MPI call. This parameter type is used
                      ///< for example in \c MPI collective operations like \c MPI_Bcast where the corresponding \c MPI
                      ///< function expects only one \c MPI_Datatype parameter of type \c MPI_Datatype.
};
/// @}
} // namespace internal
} // namespace kamping
