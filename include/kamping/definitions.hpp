/// @file
/// @brief File contain some definitions used by the KaMPI.ng library

#pragma once

namespace kamping {
namespace internal {

/// @addtogroup kamping_utility
/// @{


///@brief Each input parameter to one of the \c MPI calls wrapped by KaMPI.ng needs to has one of the following tags.
///
/// The \c MPI calls wrapped by KaMPI.ng do not rely on the restricting positional parameter paradigm but used named
/// parameters instead. The ParameterTypes defined in this enum are necessary to implement this approach, as KaMPIng
/// needs to identify the purpose of each (unordered) argument.
enum class ParameterType {
    send_buf,
    recv_buf,
    recv_counts,
    recv_displs,
    send_counts,
    send_displs,
    sender,
    receiver,
    root
};
///@}
} // namespace internal
} // namespace kamping
