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
/// @brief Some functions and types simplifying/enabling the development of wrapped \c MPI calls in KaMPI.ng.

#include <utility>

namespace kamping {
namespace internal {
// https://stackoverflow.com/a/9154394 TODO license?
/// @brief Helper to implement has_extract_v
template <typename>
struct true_type : std::true_type {};

/// @brief Helper to implement has_extract_v
template <typename T>
auto test_extract(int) -> true_type<decltype(std::declval<T>().extract())>;

/// @brief Helper to implement has_extract_v
template <typename T>
auto test_extract(...) -> std::false_type;

/// @brief Helper to implement has_extract_v
template <typename T>
struct has_extract : decltype(internal::test_extract<T>(0)) {};

/// @brief has_extract_v is \c true iff type T has a member function \c extract().
///
/// @tparam T Type which is tested for the existence of a member function.
template <typename T>
inline constexpr bool has_extract_v = has_extract<T>::value;

/// @brief Use this type if one of the template parameters of MPIResult is not used for a specific wrapped \c MPI call.
struct BufferCategoryNotUsed {};
} // namespace internal

/// @brief MPIResult contains the result of a \c MPI call wrapped by KaMPI.ng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transferred to KaMPI.ng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every wrapped \c MPI call.
/// If a specific call does not use a buffer category, you have to provide BufferCategoryNotUsed instead.
///
/// @tparam RecBuf Buffer type containing the received elements.
/// @tparam RecCounts Buffer type containing the numbers of received elements.
/// @tparam RecvCount Value wrapper type containing the number of received elements.
/// @tparam RecDispls Buffer type containing the displacements of the received elements.
/// @tparam SendDispls Buffer type containing the displacements of the sent elements.
/// @tparam MPIStatusObject Buffer type containing the \c MPI status object(s).
template <class RecvBuf, class RecvCounts, class RecvCount, class RecvDispls, class SendDispls>
class MPIResult {
public:
    /// @brief Constructor of MPIResult.
    ///
    /// If any of the buffer categories are not used by the wrapped \c MPI call or if the caller has provided (and still
    /// owns) the memory for the associated results, the empty placeholder type BufferCategoryNotUsed must be passed to
    /// the constructor instead of an actual buffer object.
    MPIResult(
        RecvBuf&& recv_buf, RecvCounts&& recv_counts, RecvCount&& recv_count, RecvDispls&& recv_displs,
        SendDispls&& send_displs)
        : _recv_buffer(std::forward<RecvBuf>(recv_buf)),
          _recv_counts(std::forward<RecvCounts>(recv_counts)),
          _recv_count(std::forward<RecvCount>(recv_count)),
          _recv_displs(std::forward<RecvDispls>(recv_displs)),
          _send_displs(std::forward<SendDispls>(send_displs)) {}

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvBuf_ Template parameter helper only needed to remove this function if RecvBuf does not possess a
    /// member function \c extract().
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
        typename RecvCounts_ = RecvCounts, std::enable_if_t<kamping::internal::has_extract_v<RecvCounts_>, bool> = true>
    decltype(auto) extract_recv_counts() {
        return _recv_counts.extract();
    }

    /// @brief Extracts the \c recv_count from the MPIResult object.
    ///
    /// This function is only available if the MPIResult object owns a recv count.
    /// @tparam RecvCount_ Template parameter helper only needed to remove this function if RecvCount does not possess a
    /// member function \c extract().
    /// @return Returns the underlying recv count.
    template <
        typename RecvCount_ = RecvCount, std::enable_if_t<kamping::internal::has_extract_v<RecvCount_>, bool> = true>
    decltype(auto) extract_recv_count() {
        return _recv_count.extract();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvDispls_ Template parameter helper only needed to remove this function if RecvDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <
        typename RecvDispls_ = RecvDispls, std::enable_if_t<kamping::internal::has_extract_v<RecvDispls_>, bool> = true>
    decltype(auto) extract_recv_displs() {
        return _recv_displs.extract();
    }

    /// @brief Extracts the \c send_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendDispls_ Template parameter helper only needed to remove this function if SendDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the send displacements.
    template <
        typename SendDispls_ = SendDispls, std::enable_if_t<kamping::internal::has_extract_v<SendDispls_>, bool> = true>
    decltype(auto) extract_send_displs() {
        return _send_displs.extract();
    }

private:
    RecvBuf _recv_buffer;    ///< Buffer object containing the received elements. May be empty if the received elements
                             ///< have been written into storage owned by the caller of KaMPI.ng.
    RecvCounts _recv_counts; ///< Buffer object containing the receive counts. May be empty if the receive counts have
                             ///< been written into storage owned by the caller of KaMPI.ng.
    RecvCount _recv_count;   ///< Object containing the receive count. May be empty if the operation does not yield a
                             ///< receive count.
    RecvDispls _recv_displs; ///< Buffer object containing the receive displacements. May be empty if the receive
                             ///< displacements have been written into storage owned by the caller of KaMPI.ng.
    SendDispls _send_displs; ///< Buffer object containing the send displacements. May be empty if the send
                             ///< displacements have been written into storage owned by the caller of KaMPI.ng.
};

} // namespace kamping
