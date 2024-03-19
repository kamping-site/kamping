// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
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
#include <tuple>
#include <utility>

#include "kamping/has_member.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_filtering.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "named_parameter_selection.hpp"

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

/// @brief Helper for implementing the extract_* functions in \ref MPIResult. Is \c true if the passed buffer type owns
/// its underlying storage and is an output buffer.
template <typename Buffer>
inline constexpr bool is_extractable = Buffer::is_owning&& Buffer::is_out_buffer;

/// @brief Specialization of helper for implementing the extract_* functions in \ref MPIResult. Is always \c false;
template <>
inline constexpr bool is_extractable<internal::ResultCategoryNotUsed> = false;
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
/// have to provide ResultCategoryNotUsed instead.
///
/// @tparam Args Types of return data buffers.
template <typename... Args>
class MPIResult {
public:
    /// @brief \c true, if the result does not encapsulate any data.
    static constexpr bool is_empty = (sizeof...(Args) == 0);
    /// @brief \c true, if the result encapsulates a recv_buf.
    static constexpr bool has_recv_buffer = internal::has_parameter_type<internal::ParameterType::recv_buf, Args...>();
    /// @brief \c true, if the result encapsulates a send_recv_buf.
    static constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
    static_assert(
        !(has_recv_buffer && has_send_recv_buffer),
        "We cannot have a recv and a send_recv buffer contained in the result object."
    );

    /// @brief Constructor for MPIResult.
    ///
    /// @param data std::tuple containing all data buffers to be returned.
    MPIResult(std::tuple<Args...>&& data) : _data(std::move(data)) {}

    /// @brief Extracts the \c kamping::Status from the MPIResult object.
    ///
    /// This function is only available if the underlying status is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying status object.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::status, T>(), bool> = true>
    decltype(auto) extract_status() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::status>(_data).extract();
    }

    /// @brief Get the \c kamping::Status from the MPIResult object.
    ///
    /// This function is only available if the underlying status is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying status object.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::status, T>(), bool> = true>
    auto& get_status() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::status>(_data).underlying();
    }

    /// @brief Get the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_buf, T>(), bool> = true>
    auto& get_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_buf>(_data).underlying();
    }

    /// @brief Get the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_buf, T>(), bool> = true>
    auto const& get_recv_buffer() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_buf>(_data).underlying();
    }

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_buf, T>(), bool> = true>
    decltype(auto) extract_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_buf>(_data).extract();
    }

    /// @brief Get the \c send_recv_buffer from the MPIResult object. @todo discuss this
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_buf, T>(), bool> =
            true>
    auto& get_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_buf>(_data).underlying();
    }

    /// @brief Get the \c send_recv_buffer from the MPIResult object. @todo discuss this
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_buf, T>(), bool> =
            true>
    auto const& get_recv_buffer() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_buf>(_data).underlying();
    }

    /// @brief Extracts the \c send_recv_buffer from the MPIResult object. @todo discuss this
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_buf, T>(), bool> =
            true>
    decltype(auto) extract_recv_buffer() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_buf>(_data).extract();
    }

    /// @brief Get the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the receive counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_counts, T>(), bool> = true>
    auto& get_recv_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_counts>(_data).underlying();
    }

    /// @brief Get the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the receive counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_counts, T>(), bool> = true>
    auto const& get_recv_counts() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_counts>(_data).underlying();
    }

    /// @brief Extracts the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the receive counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_counts, T>(), bool> = true>
    decltype(auto) extract_recv_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_counts>(_data).extract();
    }

    /// @brief Extracts the \c recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the recv count.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_count, T>(), bool> = true>
    decltype(auto) extract_recv_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_count>(_data).extract();
    }

    /// @brief Gets the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the receive displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_displs, T>(), bool> = true>
    auto& get_recv_displs() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_displs>(_data).underlying();
    }

    /// @brief Gets the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the receive displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_displs, T>(), bool> = true>
    auto const& get_recv_displs() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_displs>(_data).underlying();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_displs, T>(), bool> = true>
    decltype(auto) extract_recv_displs() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_displs>(_data).extract();
    }

    /// @brief Gets the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_counts, T>(), bool> = true>
    auto& get_send_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_counts>(_data).underlying();
    }

    /// @brief Gets the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_counts, T>(), bool> = true>
    auto const& get_send_counts() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_counts>(_data).underlying();
    }

    /// @brief Extracts the \c send_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_counts, T>(), bool> = true>
    decltype(auto) extract_send_counts() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_counts>(_data).extract();
    }

    /// @brief Gets the \c send_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_count, T>(), bool> = true>
    auto& get_send_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_count>(_data).underlying();
    }

    /// @brief Gets the \c send_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns a reference to the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_count, T>(), bool> = true>
    auto const& get_send_count() const {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_count>(_data).underlying();
    }

    /// @brief Extracts the \c send_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_count, T>(), bool> = true>
    decltype(auto) extract_send_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_count>(_data).extract();
    }

    /// @brief Extracts the \c send_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_displs, T>(), bool> = true>
    decltype(auto) extract_send_displs() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_displs>(_data).extract();
    }

    /// @brief Extracts the \c send_recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_recv_count.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_count, T>(), bool> =
            true>
    decltype(auto) extract_send_recv_count() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_count>(_data).extract();
    }

    /// @brief Extracts the \c send_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_type, T>(), bool> = true>
    decltype(auto) extract_send_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_type>(_data).extract();
    }

    /// @brief Extracts the \c recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::recv_type, T>(), bool> = true>
    decltype(auto) extract_recv_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::recv_type>(_data).extract();
    }

    /// @brief Extracts the \c send_recv_type from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam T Template parameter helper only needed to remove this
    /// function if the corresponding buffer does not exist or exists but does not possess a member function \c
    /// extract().
    /// @return Returns the underlying storage containing the send_type.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_type_in_tuple<internal::ParameterType::send_recv_type, T>(), bool> =
            true>
    decltype(auto) extract_send_recv_type() {
        return internal::select_parameter_type_in_tuple<internal::ParameterType::send_recv_type>(_data).extract();
    }

    /// @brief Get the underlying data from the i-th buffer in the result object. This method is part of the
    /// structured binding enabling machinery.
    ///
    /// @tparam i Index of the data buffer to extract.
    /// @return Returns a reference to the underlying data of the i-th data buffer.
    template <std::size_t i>
    auto& get() {
        return std::get<i>(_data).underlying();
    }

    /// @brief Get the underlying data from the i-th buffer in the result object. This method is part of the
    /// structured binding enabling machinery.
    ///
    /// @tparam i Index of the data buffer to extract.
    /// @return Returns a reference to the underlying data of the i-th data buffer.
    template <std::size_t i>
    auto const& get() const {
        return std::get<i>(_data).underlying();
    }

private:
    std::tuple<Args...> _data; ///< tuple storing the data buffers
};

/// @brief Primary template for result trait indicates whether the result object is empty.
template <typename T>
constexpr bool is_result_empty_v = false;

/// @brief Template specialization for result trait indicates whether the result object is
/// empty.
template <typename... Args>
constexpr bool is_result_empty_v<MPIResult<Args...>> = MPIResult<Args...>::is_empty;

/// @brief Template specialization for result trait indicates whether the result object is
/// empty.
template <>
inline constexpr bool is_result_empty_v<void> = true;

} // namespace kamping

namespace std {

/// @brief Specialization of the std::tuple_size for \ref kamping::MPIResult. Part of the structured binding machinery.
///
/// @tparam Args Automatically deducted template parameters.
template <typename... Args>
struct tuple_size<kamping::MPIResult<Args...>> {
    static constexpr size_t value = sizeof...(Args); ///< Number of data buffers in the \ref kamping::MPIResult.
};

/// @brief Specialization of the std::tuple_element for \ref kamping::MPIResult. Part of the structured binding
/// machinery.
///
/// @param index Index of the entry of \ref kamping::MPIResult for which the underlying data type shall be deduced.
/// @tparam Args Automatically deducted template parameters.
template <size_t index, typename... Args>
struct tuple_element<index, kamping::MPIResult<Args...>> {
    using type = std::remove_reference_t<decltype(declval<kamping::MPIResult<Args...>>().template get<index>()
    )>; ///< Type of the underlying data of the i-th data buffer in the result object.
};

} // namespace std

namespace kamping {

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
    /// If this result owns the underlying request:
    /// - returns a \c std::pair containing the \ref Request and \ref
    /// MPIResult if the result object contains owning out buffers.
    /// - returns only the \ref Request object otherwise.
    ///
    /// If the request is owned by the user
    /// - return the underlying \ref MPIResult if the result object contains any owning out buffers.
    /// - returns nothing otherwise.
    ///
    /// Note that the result may be in an undefined state because the associated operations is still underway and it
    /// is the user's responsibilty to ensure that the corresponding request has been completed before accessing the
    /// result.
    auto extract() {
        if constexpr (owns_request) {
            if constexpr (is_result_empty_v<decltype(extract_result())>) {
                return _request.extract();
            } else {
                auto result =
                    extract_result(); // we try to extract the result first, so that we get a nice error message
                // TODO: return a named struct
                return std::pair(_request.extract(), std::move(result));
            }
        } else {
            if constexpr (is_result_empty_v<decltype(extract_result())>) {
                return;
            } else {
                return extract_result();
            }
        }
    }

    /// @brief Waits for the underlying \ref Request to complete by calling \ref Request::wait() and upon completion
    /// returns:
    ///
    /// If \p status is an out-parameter:
    /// - If the result is not empty (see \ref is_result_empty_v), an \c std::pair containing an \ref MPIResult
    /// and the status.
    /// - If the result is empty, only the status is returned.
    ///
    /// If \p is \c kamping::status(ignore<>), or not an out-paramter:
    /// - If the result is not empty (see \ref is_result_empty_v), only the result is returned.
    /// - If the result is empty, nothing is returned.
    ///
    /// This method is only available if this result owns the underlying request. If this is not the case, the user
    /// must manually wait on the request that they own and manually obtain the result via \ref extract().
    ///
    /// @param status A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    template <
        typename StatusParamObjectType = decltype(status(ignore<>)),
        typename NonBlockingResulType_ = NonBlockingResult<MPIResultType, RequestDataBuffer>,
        typename std::enable_if<NonBlockingResulType_::owns_request, bool>::type = true>
    auto wait(StatusParamObjectType status = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        kassert_not_extracted("The result of this request has already been extracted.");
        constexpr bool return_status = internal::is_extractable<StatusParamObjectType>;
        if constexpr (!is_result_empty_v<MPIResultType>) {
            if constexpr (return_status) {
                auto status_return = _request.underlying().wait(std::move(status));
                return std::make_pair(extract_result(), std::move(status_return));
            } else {
                _request.underlying().wait(std::move(status));
                return extract_result();
            }
        } else {
            return _request.underlying().wait(std::move(status));
        }
    }

    /// @brief Tests the underlying \ref Request for completion by calling \ref
    /// Request::test() and returns a value convertible to \c bool indicating if the request is complete.
    ///
    /// The type of the return value depends on the encapsulated result and the \p status parameter and follows the
    /// same semantics as \ref wait(), but its return value is wrapped in an \c std::optional. The optional only
    /// contains a value if the request is complete, i.e. \c test() succeeded.
    ///
    /// If both the result is empty and no status returned, returns a \c bool indicating completion instead of an \c
    /// std::optional.
    ///
    /// This method is only available if this result owns the underlying request. If this is not the case, the user
    /// must manually test the request that they own and manually obtain the result via \ref extract().
    ///
    /// @param status A parameter created by \ref kamping::status() or \ref kamping::status_out().
    /// Defaults to \c kamping::status(ignore<>).
    template <
        typename StatusParamObjectType = decltype(status(ignore<>)),
        typename NonBlockingResulType_ = NonBlockingResult<MPIResultType, RequestDataBuffer>,
        typename std::enable_if<NonBlockingResulType_::owns_request, bool>::type = true>
    auto test(StatusParamObjectType status = kamping::status(ignore<>)) {
        static_assert(
            StatusParamObjectType::parameter_type == internal::ParameterType::status,
            "Only status parameters are allowed."
        );
        kassert_not_extracted("The result of this request has already been extracted.");
        constexpr bool return_status = internal::is_extractable<StatusParamObjectType>;
        if constexpr (!is_result_empty_v<MPIResultType>) {
            if constexpr (return_status) {
                auto status_return = _request.underlying().test(std::move(status));
                if (status_return) {
                    return std::optional{std::pair{extract_result(), std::move(*status_return)}};
                } else {
                    return std::optional<std::pair<MPIResultType, typename decltype(status_return)::value_type>>{};
                }
            } else {
                if (_request.underlying().test(std::move(status))) {
                    return std::optional{extract_result()};
                } else {
                    return std::optional<MPIResultType>{};
                }
            }
        } else {
            return _request.underlying().test(std::move(status));
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
} // namespace kamping

namespace kamping::internal {

/// @brief Determines whether only the recv (send_recv) buffer or multiple different buffers will be returned.
/// @tparam CallerProvidedOwningOutBuffers An std::tuple containing the types of the owning, out buffers explicitly
/// requested by the caller of the wrapped MPI call.
/// @returns \c True if the recv (send_recv) buffer is either not mentioned explicitly and no other (owning) out buffers
/// are requested or the only explicitly requested owning out buffer is the recv_buf. \c False otherwise.
template <typename CallerProvidedOwningOutBuffers>
constexpr bool return_recv_or_send_recv_buffer_only() {
    constexpr std::size_t num_caller_provided_owning_out_buffers = std::tuple_size_v<CallerProvidedOwningOutBuffers>;
    if constexpr (num_caller_provided_owning_out_buffers == 0) {
        return true;
    } else if constexpr (num_caller_provided_owning_out_buffers == 1 && std::tuple_element_t<0, CallerProvidedOwningOutBuffers>::value == ParameterType::recv_buf) {
        return true;
    } else if constexpr (num_caller_provided_owning_out_buffers == 1 && std::tuple_element_t<0, CallerProvidedOwningOutBuffers>::value == ParameterType::send_recv_buf) {
        return true;
    } else {
        return false;
    }
}

/// @brief Checks whether a buffer with parameter type recv_buf or a buffer with type send_recv_buf is present and
/// returns the found parameter type. Note that we require that either a recv_buf or a send_recv_buf is present.
///
/// @tparam Buffers All buffer types to be searched for type `recv_buf` or `send_recv_buf`.
/// @returns The parameter type of the first buffer whose parameter type is recv_buf or send_recv_buf.
template <typename... Buffers>
constexpr ParameterType determine_recv_buffer_type() {
    constexpr bool has_recv_buffer = internal::has_parameter_type<internal::ParameterType::recv_buf, Buffers...>();
    constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Buffers...>();
    static_assert(has_recv_buffer ^ has_send_recv_buffer, "either a recv or a send_recv buffer must be present");
    if constexpr (has_recv_buffer) {
        return ParameterType::recv_buf;
    } else {
        return ParameterType::send_recv_buf;
    }
}

/// @brief List of parameter type (entries) which should not be included in the result object.
using parameter_types_to_ignore_for_result_object = type_list<
    ParameterTypeEntry<ParameterType::op>,
    ParameterTypeEntry<ParameterType::source>,
    ParameterTypeEntry<ParameterType::destination>,
    ParameterTypeEntry<ParameterType::statuses>,
    ParameterTypeEntry<ParameterType::request>,
    ParameterTypeEntry<ParameterType::root>,
    ParameterTypeEntry<ParameterType::tag>,
    ParameterTypeEntry<ParameterType::send_mode>,
    ParameterTypeEntry<ParameterType::values_on_rank_0>>;

///@brief Predicate to check whether a buffer provided to \ref make_mpi_result() shall be discard or returned in the
/// result object.
struct PredicateForResultObject {
    ///@brief Discard function to check whether a buffer provided to \ref make_mpi_result() shall be discard or returned
    /// in the result object.
    /// call.
    ///
    ///@tparam BufferType BufferType to be checked.
    ///@return \c True (i.e. discard) iff Arg's parameter_type is `sparse_send_buf`, `on_message` or `destination`.
    template <typename BufferType>
    static constexpr bool discard() {
        using ptype_entry = ParameterTypeEntry<BufferType::parameter_type>;
        if constexpr (parameter_types_to_ignore_for_result_object::contains<ptype_entry>) {
            return true;
        } else {
            return !BufferType::is_owning
                   || !BufferType::is_out_buffer; ///< Predicate which Head has to fulfill to be kept.
        }
    }
};

/// @brief Helper to check if a type `T` has a member type `T::DataBufferType`.
template <typename, typename = void>
constexpr bool has_data_buffer_type_member = false;

/// @brief Helper to check if a type `T` has a member type `T::DataBufferType`.
template <typename T>
constexpr bool has_data_buffer_type_member<T, std::void_t<typename T::DataBufferType>> = true;

///@brief Predicate to check whether a buffer provided to \ref make_mpi_result() shall be discard or returned in the
/// result object, including a hotfix for serialization.
struct DiscardSerializationBuffers {
    /// @brief Discard function to check whether a buffer provided to \ref make_mpi_result() shall be discard or
    /// returned in the result object. call.
    ///
    /// @tparam BufferType BufferType to be checked.
    /// @return \c True (i.e. discard) iff \ref PredicateForResultObject discards this, or if the parameter uses
    /// serialization, so we don't expose serialization buffers to the user.
    ///
    /// @todo this a quick and dirty hack, in the future we want to select which parameters to return based on a flag.
    /// Currently we assume that we want to return everything that is out and owning.
    template <typename BufferType>
    static constexpr bool discard() {
        if (PredicateForResultObject::discard<BufferType>()) {
            return true;
        }
        // we sometimes call this with DataBuffers and sometimes with DataBufferBuilder, so we need a case distinction
        // here.
        using ptype_entry = ParameterTypeEntry<BufferType::parameter_type>;
        if constexpr (ptype_entry::parameter_type == internal::ParameterType::recv_buf || ptype_entry::parameter_type == internal::ParameterType::send_recv_buf) {
            if constexpr (has_data_buffer_type_member<BufferType>) {
                return buffer_uses_serialization<typename BufferType::DataBufferType>;
            } else {
                return buffer_uses_serialization<BufferType>;
            }
        }
        return false;
    }
};

/// @brief Returns True iff only a recv or send_recv buffer is present.
/// Communicator::ibarrier()).
///
/// @tparam Buffers All buffer types to be searched for type `status`.
template <typename... Buffers>
constexpr bool has_recv_or_send_recv_buf() {
    constexpr bool has_recv_buffer = internal::has_parameter_type<internal::ParameterType::recv_buf, Buffers...>();
    constexpr bool has_send_recv_buffer =
        internal::has_parameter_type<internal::ParameterType::send_recv_buf, Buffers...>();
    return has_recv_buffer || has_send_recv_buffer;
}

/// @brief Template class to prepend the ParameterTypeEntry<ParameterType::ptype> type to a given std::tuple.
/// @tparam ptype ParameterType to prepend
/// @tparam Tuple An std::tuple.
template <ParameterType ptype, typename Tuple>
struct PrependParameterType {
    using type = typename PrependType<std::integral_constant<internal::ParameterType, ptype>, Tuple>::
        type; ///< Concatenated tuple, i.e. type = std::tuple<TypeToPrepend, (Type contained in Tuple)... >.
};

/// @brief Construct result object for a wrapped MPI call. Four different cases are handled:
/// a) The recv_buffer owns its underlying data (i.e. the received data has to be returned via the result object):
///
/// a.1) The recv_buffer is the only buffer to be returned, i.e. the only caller provided owning out buffer:
/// In this case, the recv_buffers's underlying data is extracted and returned directly (by value).
///
/// a.2) There are multiple buffers to be returned and recv_buffer is explicitly provided by the caller:
/// In this case a \ref kamping::MPIResult object is created, which stores the buffers to return (owning out buffers)
/// in a std::tuple respecting the order in which these buffers where provided to the wrapped MPI call. This enables
/// unpacking the object via structured binding.
///
/// a.3) There are more data buffers to be returned and recv_buffer is *not* explicitly provided by the caller:
/// In this case a \ref kamping::MPIResult object is created, which stores the buffers to return. The
/// recv_buffer is always the first entry in the result object followed by the other buffers respecting the order in
/// which these buffers where provided to the wrapped MPI call.
///
/// b) There is no recv buffer (see \ref Communicator::probe() for example) or the recv_buffer only references its
/// underlying data (i.e. it is a non-owning out buffer): In this case recv_buffer is not part of the result object. The
/// \ref kamping::MPIResult object stores the buffer to return (owning buffers for which a *_out() named parameter was
/// passed to the wrapped MPI call) in a std::tuple respecting the order in which these buffers where provided to the
/// wrapped MPI call.
///
/// @tparam CallerProvidedArgs Types of arguments passed to the wrapped MPI call.
/// @tparam Buffers Types of data buffers created/filled within the wrapped MPI call.
/// @param buffers data buffers created/filled within the wrapped MPI call.
/// @return result object as specified above.
///
/// @see \ref docs/named_parameters.md
template <typename CallerProvidedArgs, typename... Buffers>
auto make_mpi_result(Buffers&&... buffers) {
    // filter named parameters provided to the wrapped MPI function and keep only owning out parameters (=owning out
    // buffers)
    using CallerProvidedOwningOutParameters =
        typename internal::FilterOut<PredicateForResultObject, CallerProvidedArgs>::type;
    using CallerProvidedOwningOutParametersWithoutSerializationBuffers =
        typename internal::FilterOut<DiscardSerializationBuffers, CallerProvidedArgs>::type;
    constexpr std::size_t num_caller_provided_owning_out_buffers =
        std::tuple_size_v<CallerProvidedOwningOutParametersWithoutSerializationBuffers>;
    if constexpr (!has_recv_or_send_recv_buf<Buffers...>()) {
        // do no special handling for receive buffer at all, since there is none.
        return MPIResult(construct_buffer_tuple<CallerProvidedOwningOutParameters>(buffers...));
    } else {
        // receive (send-receive) buffer needs (potentially) a special treatment (if it is an owning (out) buffer
        // and provided by the caller)
        constexpr internal::ParameterType recv_parameter_type = determine_recv_buffer_type<Buffers...>();
        auto&          recv_or_send_recv_buffer = internal::select_parameter_type<recv_parameter_type>(buffers...);
        constexpr bool recv_or_send_recv_buf_is_owning =
            std::remove_reference_t<decltype(recv_or_send_recv_buffer)>::is_owning;
        constexpr bool recv_or_send_recv_buffer_is_owning_and_provided_by_caller =
            has_parameter_type_in_tuple<recv_parameter_type, CallerProvidedOwningOutParameters>();

        // special case 1: recv (send_recv) buffer is not owning
        if constexpr (!recv_or_send_recv_buf_is_owning) {
            if constexpr (num_caller_provided_owning_out_buffers == 0) {
                // there are no buffers to return
                return;
            } else {
                // no special treatment of recv buffer is needed as the recv_buffer is not part of the result
                // object anyway.
                return MPIResult(construct_buffer_tuple<CallerProvidedOwningOutParameters>(buffers...));
            }
        }
        // specialcase 2: recv (send_recv) buffer is the only owning out parameter
        else if constexpr (return_recv_or_send_recv_buffer_only<CallerProvidedOwningOutParameters>()) {
            // if only the receive buffer shall be returned, its underlying data is returned directly instead of a
            // wrapping result object
            return recv_or_send_recv_buffer.extract();
        }

        // case A: recv (send_recv) buffer is provided by caller (and owning)
        else if constexpr (recv_or_send_recv_buffer_is_owning_and_provided_by_caller) {
            return MPIResult(construct_buffer_tuple<CallerProvidedOwningOutParameters>(buffers...));
        }
        // case B: recv buffer is not provided by caller -> recv buffer will be stored as first entry in
        // underlying result object
        else {
            using ParametersToReturn =
                typename PrependParameterType<recv_parameter_type, CallerProvidedOwningOutParameters>::type;
            return MPIResult(construct_buffer_tuple<ParametersToReturn>(buffers...));
        }
    }
}

/// @brief Factory for creating a \ref kamping::NonBlockingResult.
///
/// Makes an \ref kamping::NonBlockingResult from all arguments passed and inserts internal::ResultCategoryNotUsed
/// when no fitting parameter type is passed as argument.
///
/// Note that an argument of with type \ref kamping::internal::ParameterType::request is required.
///
/// @tparam CallerProvidedArgs Types of arguments passed to the wrapped MPI call.
/// @tparam Args Automatically deducted template parameters.
/// @param args All parameter that should be included in the MPIResult.
/// @return \ref kamping::NonBlockingResult encapsulating all passed parameters.
template <typename CallerProvidedArgsInTuple, typename... Args>
auto make_nonblocking_result(Args... args) {
    auto&& request                 = internal::select_parameter_type<internal::ParameterType::request>(args...);
    auto   construct_result_object = [&]() {
        return make_mpi_result<CallerProvidedArgsInTuple>(std::forward<Args>(args)...);
    };
    using result_type = decltype(construct_result_object());
    if constexpr (is_result_empty_v<result_type>) {
        return NonBlockingResult(MPIResult{std::tuple<>{}}, std::move(request));
    } else {
        return NonBlockingResult(construct_result_object(), std::move(request));
    }
}

} // namespace kamping::internal
