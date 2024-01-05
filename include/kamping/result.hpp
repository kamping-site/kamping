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
// <https://www.gnu.org/licenses/>.:

#pragma once

/// @file
/// @brief Some functions and types simplifying/enabling the development of wrapped \c MPI calls in KaMPIng.

#include <optional>
#include <utility>

#include "kamping/has_member.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
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
struct ResultCategoryNotUsed {};
} // namespace internal

/// @brief Helper for implementing the extract_* functions. Is \c true if the passed buffer type owns its
/// underlying storage and is an output buffer.
template <typename Buffer>
inline constexpr bool is_extractable = Buffer::is_owning& Buffer::is_out_buffer;

/// @brief Specialization of helper for implementing the extract_* functions. Is always \c false;
template <>
inline constexpr bool is_extractable<internal::ResultCategoryNotUsed> = false;

/// @brief MPIResult contains the result of a \c MPI call wrapped by KaMPIng.
///
/// A wrapped \c MPI call can have multiple different results such as the \c
/// recv_buffer, \c recv_counts, \c recv_displs etc. If the buffers where these
/// results have been written to by the library call has been allocated
/// by/transferred to KaMPIng, the content of the buffers can be extracted using
/// extract_<result>.
/// Note that not all below-listed buffer categories needs to be used by every
/// wrapped \c MPI call. If a specific call does not use a buffer category, you
/// have to provide ResultCategoryNotUsed instead.
///
/// @tparam StatusObject Buffer type containing the \c MPI status object(s).
/// @tparam RecvBuf Buffer type containing the received elements.
/// @tparam RecvCounts Buffer type containing the numbers of received elements.
/// @tparam RecvDispls Buffer type containing the displacements of the received
/// elements.
/// @tparam SendDispls Buffer type containing the displacements of the sent
/// elements.
/// @tparam SendRecvCount Buffer type containing the send recv count (only used by bcast).
template <
    class StatusObject,
    class RecvBuf,
    class RecvCounts,
    class RecvCount,
    class RecvDispls,
    class SendCounts,
    class SendCount,
    class SendDispls,
    class SendRecvCount,
    class SendType,
    class RecvType,
    class SendRecvType>
class MPIResult {
private:
    /// @brief Helper for implementing \ref is_empty. Returns \c true if all template arguments passed are equal to \ref
    /// internal::ResultCategoryNotUsed.
    template <typename... Args>
    static constexpr bool is_empty_impl = std::conjunction_v<std::is_same<Args, internal::ResultCategoryNotUsed>...>;

public:
    /// @brief \c true, if the result does not encapsulate any data.
    static constexpr bool is_empty = is_empty_impl<
        StatusObject,
        RecvBuf,
        RecvCounts,
        RecvCount,
        RecvDispls,
        SendCounts,
        SendCount,
        SendDispls,
        SendRecvCount,
        SendType,
        RecvType,
        SendRecvType>;

    /// @brief Constructor of MPIResult.
    ///
    /// If any of the buffer categories are not used by the wrapped \c MPI call or if the caller has provided (and still
    /// owns) the memory for the associated results, the empty placeholder type ResultCategoryNotUsed must be passed to
    /// the constructor instead of an actual buffer object.
    MPIResult(
        StatusObject&&  status,
        RecvBuf&&       recv_buf,
        RecvCounts&&    recv_counts,
        RecvCount&&     recv_count,
        RecvDispls&&    recv_displs,
        SendCounts&&    send_counts,
        SendCount&&     send_count,
        SendDispls&&    send_displs,
        SendRecvCount&& send_recv_count,
        SendType&&      send_type,
        RecvType&&      recv_type,
        SendRecvType&&  send_recv_type
    )
        : _status(std::forward<StatusObject>(status)),
          _recv_buffer(std::forward<RecvBuf>(recv_buf)),
          _recv_counts(std::forward<RecvCounts>(recv_counts)),
          _recv_count(std::forward<RecvCount>(recv_count)),
          _recv_displs(std::forward<RecvDispls>(recv_displs)),
          _send_counts(std::forward<SendCounts>(send_counts)),
          _send_count(std::forward<SendCount>(send_count)),
          _send_displs(std::forward<SendDispls>(send_displs)),
          _send_recv_count(std::forward<SendRecvCount>(send_recv_count)),
          _send_type(std::forward<SendType>(send_type)),
          _recv_type(std::forward<RecvType>(recv_type)),
          _send_recv_type(std::forward<SendRecvType>(send_recv_type)) {}

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
    decltype(auto) extract_status() {
        return _status.extract();
    }

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam RecvBuf_ Template parameter helper only needed to remove this
    /// function if RecvBuf does not possess a member function \c extract().
    /// @return Returns the underlying storage containing the received elements.
    template <typename RecvBuf_ = RecvBuf, std::enable_if_t<is_extractable<RecvBuf_>, bool> = true>
    decltype(auto) extract_recv_buffer() {
        return _recv_buffer.extract();
    }

    /// @brief Extracts the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvCounts_ Template parameter helper only needed to remove this function if RecvCounts does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive counts.
    template <typename RecvCounts_ = RecvCounts, std::enable_if_t<is_extractable<RecvCounts_>, bool> = true>
    decltype(auto) extract_recv_counts() {
        return _recv_counts.extract();
    }

    /// @brief Extracts the \c recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvCount_ Template parameter helper only needed to remove this function if RecvCount does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the recv count.
    template <typename RecvCount_ = RecvCount, std::enable_if_t<is_extractable<RecvCount_>, bool> = true>
    decltype(auto) extract_recv_count() {
        return _recv_count.extract();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvDispls_ Template parameter helper only needed to remove this function if RecvDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <typename RecvDispls_ = RecvDispls, std::enable_if_t<is_extractable<RecvDispls_>, bool> = true>
    decltype(auto) extract_recv_displs() {
        return _recv_displs.extract();
    }

    /// @brief Extracts the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendCounts_ Template parameter helper only needed to remove this function if SendCounts does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the send counts.
    template <typename SendCounts_ = SendCounts, std::enable_if_t<is_extractable<SendCounts_>, bool> = true>
    decltype(auto) extract_send_counts() {
        return _send_counts.extract();
    }

    /// @brief Extracts the \c send_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendCount_ Template parameter helper only needed to remove this function if SendCount does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the send count.
    template <typename SendCount_ = SendCount, std::enable_if_t<is_extractable<SendCount_>, bool> = true>
    decltype(auto) extract_send_count() {
        return _send_count.extract();
    }

    /// @brief Extracts the \c send_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendDispls_ Template parameter helper only needed to remove this function if SendDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the send displacements.
    template <typename SendDispls_ = SendDispls, std::enable_if_t<is_extractable<SendDispls_>, bool> = true>
    decltype(auto) extract_send_displs() {
        return _send_displs.extract();
    }

    /// @brief Extracts the \c send_recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendRecvCount_ Template parameter helper only needed to remove this function if SendRecvCount does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the send_recv_count.
    template <typename SendRecvCount_ = SendRecvCount, std::enable_if_t<is_extractable<SendRecvCount_>, bool> = true>
    decltype(auto) extract_send_recv_count() {
        return _send_recv_count.extract();
    }

    /// @brief Extracts the \c send_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendType_ Template parameter helper only needed to remove this function if SendType does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the send_type.
    template <typename SendType_ = SendType, std::enable_if_t<is_extractable<SendType_>, bool> = true>
    decltype(auto) extract_send_type() {
        return _send_type.extract();
    }

    /// @brief Extracts the \c recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvType_ Template parameter helper only needed to remove this function if RecvType does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the send_type.
    template <typename RecvType_ = RecvType, std::enable_if_t<is_extractable<RecvType_>, bool> = true>
    decltype(auto) extract_recv_type() {
        return _recv_type.extract();
    }
    /// @brief Extracts the \c send_recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam SendRecvType_ Template parameter helper only needed to remove this function if RecvType does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the send_type.
    template <typename SendRecvType_ = SendRecvType, std::enable_if_t<is_extractable<SendRecvType_>, bool> = true>
    decltype(auto) extract_send_recv_type() {
        return _send_recv_type.extract();
    }

private:
    StatusObject _status;    ///< The status object. May be empty if the status is owned by the caller of KaMPIng.
    RecvBuf _recv_buffer;    ///< Buffer object containing the received elements. May be empty if the received elements
                             ///< have been written into storage owned by the caller of KaMPIng.
    RecvCounts _recv_counts; ///< Buffer object containing the receive counts. May be empty if the receive counts have
                             ///< been written into storage owned by the caller of KaMPIng.
    RecvCount _recv_count;   ///< Buffer object containing the (single) receive count. May be empty if the receive count
                             ///< has been written into storage owned by the caller of KaMPIng.
    RecvDispls _recv_displs; ///< Buffer object containing the receive displacements. May be empty if the receive
                             ///< displacements have been written into storage owned by the caller of KaMPIng.
    SendCounts _send_counts; ///< Buffer object containing the send counts. May be empty if the send counts have been
                             ///< written into storage owned by the caller of KaMPIng.
    SendCount _send_count;   ///< Buffer object containing the (single) send count. May be empty if the send count has
                             ///< been written into storage owned by the caller of KaMPIng.
    SendDispls _send_displs; ///< Buffer object containing the send displacements. May be empty if the send
                             ///< displacements have been written into storage owned by the caller of KaMPIng.
    SendRecvCount _send_recv_count; ///< Buffer object containing the combined send recv count (used by bcast,
                                    ///< (ex)scan, ...). May be empty if the send recv count has been written into
                                    ///< storage owned by the caller of KaMPIng.
    SendType _send_type;            ///< Buffer object containing the send type.
                                    ///< May be empty if the send type has been written into
                                    ///< storage owned by the caller of KaMPIng.
    RecvType _recv_type;            ///< Buffer object containing the recv type.
                                    ///< May be empty if the recv type has been written into
                                    ///< storage owned by the caller of KaMPIng.
    SendRecvType _send_recv_type;   ///< Buffer object containing the send_recv_type (used by bcast, (ex)scan, ...).
                                    ///< May be empty if the recv type has been written into
                                    ///< storage owned by the caller of KaMPIng.
};

/// @brief Factory creating the MPIResult.
///
/// Makes an MPIResult from all arguments passed and inserts internal::ResultCategoryNotUsed when no fitting parameter
/// type is passed as argument.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All parameter that should be included in the MPIResult.
/// @return MPIResult encapsulating all passed parameters.
template <typename... Args>
auto make_mpi_result(Args... args) {
    using default_type = decltype(internal::ResultCategoryNotUsed{});

    static_assert(
        !(internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()
          && internal::has_parameter_type<internal::ParameterType::recv_buf, Args...>()),
        "Cannot have recv_buf and send_recv_buf at the same time."
    );
    auto&& recv_buf = [&]() {
        // I'm not sure why return value optimization doesn't apply here, but the moves seem to be necessary.
        if constexpr (internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()) {
            auto& param = internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...);
            return std::move(param);
        } else {
            auto&& param = internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_type>(
                std::tuple(),
                args...
            );
            return std::move(param);
        }
    }();

    auto&& recv_counts = internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_type>(
        std::tuple(),
        args...
    );
    auto&& recv_count = internal::select_parameter_type_or_default<internal::ParameterType::recv_count, default_type>(
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
    auto&& send_count = internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_type>(
        std::tuple(),
        args...
    );
    auto&& send_displs = internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_type>(
        std::tuple(),
        args...
    );
    auto&& send_recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_count, default_type>(
            std::tuple(),
            args...
        );
    auto&& send_type = internal::select_parameter_type_or_default<internal::ParameterType::send_type, default_type>(
        std::tuple(),
        args...
    );
    auto&& recv_type = internal::select_parameter_type_or_default<internal::ParameterType::recv_type, default_type>(
        std::tuple(),
        args...
    );
    auto&& send_recv_type =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_type, default_type>(
            std::tuple(),
            args...
        );

    auto&& status = internal::select_parameter_type_or_default<internal::ParameterType::status, default_type>(
        std::tuple(),
        args...
    );

    return MPIResult(
        std::move(status),
        std::move(recv_buf),
        std::move(recv_counts),
        std::move(recv_count),
        std::move(recv_displs),
        std::move(send_counts),
        std::move(send_count),
        std::move(send_displs),
        std::move(send_recv_count),
        std::move(send_type),
        std::move(recv_type),
        std::move(send_recv_type)
    );
}

/// @brief NonBlockingResult contains the result of a non-blocking \c MPI call wrapped by KaMPIng. It encapsulates a
/// \ref kamping::MPIResult and a \ref kamping::Request.
///
///
/// @tparam MPIResultType The underlying result type.
/// @tparam RequestDataBuffer Container encapsulating the underlying request.
template <typename MPIResultType, typename RequestDataBuffer>
class NonBlockingResult {
public:
    /// @brief Constructor for \c NonBlockingResult.
    /// @param result The underlying \ref kamping::MPIResult.
    /// @param request A \ref kamping::internal::DataBuffer containing the associated \ref kamping::Request.
    NonBlockingResult(MPIResultType result, RequestDataBuffer request)
        : _mpi_result(std::move(result)),
          _request(std::move(request)) {}

    /// @brief \c true if the result object owns the underlying \ref kamping::Request.
    static constexpr bool owns_request = internal::has_extract_v<RequestDataBuffer>;

    /// @brief Extracts the components of this results, leaving the user responsible.
    ///
    /// If this result owns the underlying request, returns a \c std::tuple containing the \ref Request and \ref
    /// MPIResult. If the request is owned by the user, just return the underlying \ref MPIResult.
    ///
    /// Note that the result may be in an undefined state because the associated operations is still underway and it is
    /// the user's responsibilty to ensure that the corresponding request has been completed before accessing the
    /// result.
    auto extract() {
        if constexpr (owns_request) {
            auto result = extract_result(); // we try to extract the result first, so that we get a nice error message
            return std::make_tuple(_request.extract(), std::move(result));
        } else {
            return extract_result();
        }
    }

    /// @brief Waits for the underlying \ref Request to complete by calling \ref Request::wait() and returns an \ref
    /// MPIResult upon completion or nothing if the result is empty (see \ref MPIResult::is_empty).
    ///
    /// This method is only available if this result owns the underlying request. If this is not the case, the user must
    /// manually wait on the request that they own and manually obtain the result via \ref extract().
    template <
        typename NonBlockingResulType_ = NonBlockingResult<MPIResultType, RequestDataBuffer>,
        typename std::enable_if<NonBlockingResulType_::owns_request, bool>::type = true>
    [[nodiscard]] std::conditional_t<!MPIResultType::is_empty, MPIResultType, void> wait() {
        kassert_not_extracted("The result of this request has already been extracted.");
        _request.underlying().wait();
        if constexpr (!MPIResultType::is_empty) {
            return extract_result();
        } else {
            return;
        }
    }

    /// @brief Tests the underlying \ref Request for completion by calling \ref Request::test() and returns an optional
    /// containing the underlying \ref MPIResult on success. If the associated operation has not completed yet, returns
    /// \c std::nullopt.
    ///
    /// Returns a \c bool indicated if the test succeeded in case the result is empty (see \ref MPIResult::is_empty).
    ///
    /// This method is only available if this result owns the underlying request. If this is not the case, the user must
    /// manually test the request that they own and manually obtain the result via \ref extract().
    template <
        typename NonBlockingResulType_ = NonBlockingResult<MPIResultType, RequestDataBuffer>,
        typename std::enable_if<NonBlockingResulType_::owns_request, bool>::type = true>
    auto test() {
        kassert_not_extracted("The result of this request has already been extracted.");
        if constexpr (!MPIResultType::is_empty) {
            if (_request.underlying().test()) {
                return std::optional{extract_result()};
            } else {
                return std::optional<MPIResultType>{};
            }
        } else {
            return _request.underlying().test();
        }
    }

private:
    /// @brief Moves the wrapped \ref MPIResult out of this object.
    MPIResultType extract_result() {
        kassert_not_extracted("The result of this request has already been extracted.");
        auto extracted = std::move(_mpi_result);
        set_extracted();
        return extracted;
    }

    void set_extracted() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        is_extracted = true;
#endif
    }

    /// @brief Throws an assertion if the extracted flag is set, i.e. the underlying status has been moved out.
    ///
    /// @param message The message for the assertion.
    void kassert_not_extracted(std::string const message [[maybe_unused]]) const {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!is_extracted, message, assert::normal);
#endif
    }
    MPIResultType     _mpi_result; ///< The wrapped \ref MPIResult.
    RequestDataBuffer _request;    ///< DataBuffer containing the wrapped \ref Request.
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    bool is_extracted = false; ///< Has the status been extracted and is therefore in an invalid state?
#endif
};

/// @brief Factory for creating a \ref kamping::NonBlockingResult.
///
/// Makes an \ref kamping::NonBlockingResult from all arguments passed and inserts internal::ResultCategoryNotUsed when
/// no fitting parameter type is passed as argument.
///
/// Note that an argument of with type \ref kamping::internal::ParameterType::request is required.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All parameter that should be included in the MPIResult.
/// @return \ref kamping::NonBlockingResult encapsulating all passed parameters.
template <typename... Args>
auto make_nonblocking_result(Args... args) {
    auto&& request = internal::select_parameter_type<internal::ParameterType::request>(args...);
    auto   result  = make_mpi_result(std::forward<Args>(args)...);
    return NonBlockingResult(std::move(result), std::move(request));
}

} // namespace kamping
