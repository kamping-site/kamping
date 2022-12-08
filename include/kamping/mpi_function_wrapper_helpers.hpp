// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:

#pragma once

/// @file
/// @brief Some functions and types simplifying/enabling the development of wrapped \c MPI calls in KaMPIng.

#include <utility>

#include "kamping/has_member.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"

namespace kamping {
namespace internal {

KAMPING_MAKE_HAS_MEMBER(extract)
/// @brief has_extract_v is \c true iff type T has a member function \c extract().
///
/// @tparam T Type which is tested for the existence of a member function.
template <typename T>
inline constexpr bool has_extract_v = has_member_extract_v<T>;

/// @brief Use this type if one of the template parameters of MPIResult is not used for a specific wrapped \c MPI call.
struct BufferCategoryNotUsed {};
} // namespace internal

/// @brief MPIResult contains the result of a \c MPI call wrapped by KaMPIng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transferred to KaMPIng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every
/// wrapped \c MPI call. If a specific call does not use a buffer category, you
/// have to provide BufferCategoryNotUsed instead.
///
/// @tparam StatusObject Buffer type containing the \c MPI status object(s).
/// @tparam RecvBuf Buffer type containing the received elements.
/// @tparam RecvCounts Buffer type containing the numbers of received elements.
/// @tparam RecvDispls Buffer type containing the displacements of the received
/// elements.
/// @tparam SendDispls Buffer type containing the displacements of the sent
/// elements.
template <class StatusObject, class RecvBuf, class RecvCounts, class RecvDispls, class SendCounts, class SendDispls>
class MPIResult {
public:
    /// @brief Constructor of MPIResult.
    ///
    /// If any of the buffer categories are not used by the wrapped \c MPI call or if the caller has provided (and still
    /// owns) the memory for the associated results, the empty placeholder type BufferCategoryNotUsed must be passed to
    /// the constructor instead of an actual buffer object.
    MPIResult(
        StatusObject&& status,
        RecvBuf&&      recv_buf,
        RecvCounts&&   recv_counts,
        RecvDispls&&   recv_displs,
        SendCounts&&   send_counts,
        SendDispls&&   send_displs
    )
        : _status(std::forward<StatusObject>(status)),
          _recv_buffer(std::forward<RecvBuf>(recv_buf)),
          _recv_counts(std::forward<RecvCounts>(recv_counts)),
          _recv_displs(std::forward<RecvDispls>(recv_displs)),
          _send_counts(std::forward<SendCounts>(send_counts)),
          _send_displs(std::forward<SendDispls>(send_displs)) {}

    /// @brief Extracts the \c kamping::Status from the MPIResult object.
    ///
    /// This function is only available if the underlying status is owned by the
    /// MPIResult object.
    /// @tparam StatusType_ Template parameter helper only needed to remove this
    /// function if StatusType does not possess a member function \c extract().
    /// @return Returns the underlying status object.
    template <
        typename StatusObject_                                                  = StatusObject,
        std::enable_if_t<kamping::internal::has_extract_v<StatusObject_>, bool> = true>
    decltype(auto) status() {
        return _status.extract();
    }

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam RecvBuf_ Template parameter helper only needed to remove this
    /// function if RecvBuf does not possess a member function \c extract().
    /// @return Returns the underlying storage containing the received elements.
    template <typename RecvBuf_ = RecvBuf, std::enable_if_t<kamping::internal::has_extract_v<RecvBuf_>, bool> = true>
    decltype(auto) extract_recv_buffer() {
        return _recv_buffer.extract();
    }

    /// @brief Extracts the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvCounts_ Template parameter helper only needed to remove this function if RecvCounts does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive counts.
    template <
        typename RecvCounts_                                                  = RecvCounts,
        std::enable_if_t<kamping::internal::has_extract_v<RecvCounts_>, bool> = true>
    decltype(auto) extract_recv_counts() {
        return _recv_counts.extract();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvDispls_ Template parameter helper only needed to remove this function if RecvDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <
        typename RecvDispls_                                                  = RecvDispls,
        std::enable_if_t<kamping::internal::has_extract_v<RecvDispls_>, bool> = true>
    decltype(auto) extract_recv_displs() {
        return _recv_displs.extract();
    }

    /// @brief Extracts the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendCounts_ Template parameter helper only needed to remove this function if SendCounts does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the send counts.
    template <
        typename SendCounts_                                                  = SendCounts,
        std::enable_if_t<kamping::internal::has_extract_v<SendCounts_>, bool> = true>
    decltype(auto) extract_send_counts() {
        return _send_counts.extract();
    }

    /// @brief Extracts the \c send_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendDispls_ Template parameter helper only needed to remove this function if SendDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the send displacements.
    template <
        typename SendDispls_                                                  = SendDispls,
        std::enable_if_t<kamping::internal::has_extract_v<SendDispls_>, bool> = true>
    decltype(auto) extract_send_displs() {
        return _send_displs.extract();
    }

private:
    StatusObject _status;    ///< The status object. May be empty if the status is owned by the caller of KaMPIng.
    RecvBuf _recv_buffer;    ///< Buffer object containing the received elements. May be empty if the received elements
                             ///< have been written into storage owned by the caller of KaMPIng.
    RecvCounts _recv_counts; ///< Buffer object containing the receive counts. May be empty if the receive counts have
                             ///< been written into storage owned by the caller of KaMPIng.
    RecvDispls _recv_displs; ///< Buffer object containing the receive displacements. May be empty if the receive
                             ///< displacements have been written into storage owned by the caller of KaMPIng.
    SendCounts _send_counts; ///< Buffer object containing the send counts. May be empty if the send counts have been
                             ///< written into storage owned by the caller of KaMPIng.
    SendDispls _send_displs; ///< Buffer object containing the send displacements. May be empty if the send
                             ///< displacements have been written into storage owned by the caller of KaMPIng.
};

/// @brief Factory creating the MPIResult.
///
/// Makes an MPIResult from all arguments passed and inserts internal::BufferCategoryNotUsed when no fitting parameter
/// type is passed as argument.
///
/// @tparam Args Automaticcaly deducted template parameters.
/// @param args All parameter that should be included in the MPIResult.
/// @return MPIResult encapsulating all passed parameters.
template <typename... Args>
auto make_mpi_result(Args... args) {
    using default_type = decltype(internal::BufferCategoryNotUsed{});

    auto&& recv_buf = internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_type>(
        std::tuple(),
        args...
    );
    auto&& recv_counts = internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_type>(
        std::tuple(),
        args...
    );
    auto&& recv_displs = internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_type>(
        std::tuple(),
        args...
    );
    auto&& send_counts = internal::select_parameter_type_or_default<internal::ParameterType::send_counts, default_type>(
        std::tuple(),
        args...
    );
    auto&& send_displs = internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_type>(
        std::tuple(),
        args...
    );

    using default_status_type = decltype(kamping::status(kamping::ignore<>));

    auto&& status = internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_type>(
        std::tuple(),
        args...
    );

    return MPIResult(
        std::move(status),
        std::move(recv_buf),
        std::move(recv_counts),
        std::move(recv_displs),
        std::move(send_counts),
        std::move(send_displs)
    );
}

} // namespace kamping
