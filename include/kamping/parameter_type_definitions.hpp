// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief File containing the parameter types used by the KaMPI.ng library

#pragma once

namespace kamping {
/// @brief Internal namespace marking the code that is not user-facing.
///
namespace internal {

/// @addtogroup kamping_utility
/// @{


/// @brief Each input parameter to one of the \c MPI calls wrapped by KaMPI.ng needs to has one of the following tags.
///
/// The \c MPI calls wrapped by KaMPI.ng do not rely on the restricting positional parameter paradigm but use named
/// parameters instead. The ParameterTypes defined in this enum are necessary to implement this approach, as KaMPI.ng
/// needs to identify the purpose of each (unordered) argument.
/// Note that not all enum entries are necessary in each wrapped \c MPI call.
enum class ParameterType {
    send_buf, ///< Tag used to represent a send buffer, i.e. a buffer containing the data elements to be sent via \c
              ///< MPI.
    recv_buf, ///< Tag used to represent a receive buffer, i.e. a buffer containing the data elements to be received via
              ///< \c MPI.
    send_recv_buf, ///< Tag used to represent a send and receive buffer, i.e. a buffer containing the data elements to
                   ///< be sent or received (depending on the process' rank) via \c MPI.
    recv_counts,   ///< Tag used to represent a receive counts buffer, i.e. a buffer containing the receive counts from
                   ///< the involved PEs.
    recv_displs,   ///< Tag used to represent a receive displacements buffer, i.e. a buffer containing the receive
                   ///< displacements from the involved PEs.
    send_counts,   ///< Tag used to represent a send counts buffer, i.e. a buffer containing the send counts from the
                   ///< involved PEs.
    send_displs, ///< Tag used to represent a send displacements buffer, i.e. a buffer containing the send displacements
                 ///< from the involved PEs.
    sender,      ///< Tag used to represent the sending PE in a \c MPI call.
    op,          ///< Tag used to represent a reduce operation in a \c MPI call.
    receiver,    ///< Tag used to represent the receiving PE in a \c MPI call.
    root         ///< Tag used to represent the root PE in a \c MPI collectives call.
};
/// @}
} // namespace internal
} // namespace kamping
