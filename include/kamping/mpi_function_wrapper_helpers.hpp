// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
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
template <typename>
struct true_type : std::true_type {};
template <typename T>
auto test_extract(int) -> true_type<decltype(std::declval<T>().extract())>;
template <typename T>
auto test_extract(...) -> std::false_type;
template <typename T>
struct has_extract : decltype(internal::test_extract<T>(0)) {};
template <typename T>
inline constexpr bool has_extract_v = has_extract<T>::value;


///@brief Use this type if one of the template parameters of MPIResult is not used for a specific wrapped \c MPI call.
struct BufferCategoryNotUsed {};
} // namespace internal

///@brief MPIResult contains the result of a \c MPI call wrapped by KaMPI.ng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transfered to KaMPI.ng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every wrapped \c MPI call.
/// If a specific call does not use a buffer category, you have to provide internal::BufferCategoryNotUsed instead.
///
///@tparam RecBuf Buffer type containing the received elements.
///@tparam RecCounts Buffer type containing the numbers of received elements.
///@tparam RecDispls Buffer type containing the displacements of the received elements.
///@tparam SendDispls Buffer type containing the displacements of the sent elements.
///@tparam MPIStatusObject Buffer type containing the \c MPI status object(s).
template <class RecvBuf, class RecvCounts, class RecvDispls, class SendDispls, class MPIStatusObject>
class MPIResult {
public:
    MPIResult(
        RecvBuf&& recv_buf, RecvCounts&& recv_counts, RecvDispls&& recv_displs, SendDispls&& send_displs,
        MPIStatusObject&& mpi_status)
        : _recv_buffer(std::forward<RecvBuf>(recv_buf)),
          _recv_counts(std::forward<RecvCounts>(recv_counts)),
          _recv_displs(std::forward<RecvDispls>(recv_displs)),
          _send_displs(std::forward<SendDispls>(send_displs)),
          _mpi_status(std::forward<MPIStatusObject>(mpi_status)) {}

    template <typename U = RecvBuf, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_buffer() {
        return _recv_buffer.extract();
    }

    template <typename U = RecvCounts, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_counts() {
        return _recv_counts.extract();
    }

    template <typename U = RecvDispls, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_recv_displs() {
        return _recv_displs.extract();
    }

    template <typename U = SendDispls, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_send_displs() {
        return _send_displs.extract();
    }

    template <typename U = MPIStatusObject, std::enable_if_t<kamping::internal::has_extract_v<U>, bool> = true>
    decltype(auto) extract_mpi_status() {
        return _mpi_status.extract();
    }

private:
    RecvBuf         _recv_buffer;
    RecvCounts      _recv_counts;
    RecvDispls      _recv_displs;
    SendDispls      _send_displs;
    MPIStatusObject _mpi_status;
};

} // namespace kamping
