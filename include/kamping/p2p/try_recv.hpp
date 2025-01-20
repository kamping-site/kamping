// This file is part of KaMPIng.
//
// Copyright 2023-2024 The KaMPIng Authors
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

#include <type_traits>
#include <utility>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/helpers.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "kamping/status.hpp"

///// @addtogroup kamping_p2p
/// @{

// @brief Receives a message if one is available.
///
/// In contrast to \ref kamping::Communicator::recv(), this method does not block if no message is available. Instead,
/// it will return a empty \c std::optional. Internally, this first does a matched probe (\c MPI_Improbe) to check if a
/// message is available. If a message is available, it will be received using a matched receive (\c MPI_Mrecv).
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() the buffer to receive the message into. The buffer's underlying storage must be large
/// enough to hold all received elements. If no \ref kamping::recv_buf() is provided, the \c value_type of the recv
/// buffer has to be passed as a template parameter to \c recv().
///
/// - \ref kamping::tag() receive message with the given tag. Defaults to receiving for an arbitrary tag, i.e. \c
/// tag(tags::any).
///
/// - \ref kamping::source() receive a message sent from the given source rank. Defaults to probing for an arbitrary
/// source, i.e. \c source(rank::any).
///
/// - \ref kamping::status(). Returns info about the received message by setting the
/// appropriate fields in the status object. The status can be obtained by using \c kamping::status_out and ignored by
/// passing \c kamping::ignore<>. This is the default.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as the recv type. If omitted, the \c MPI datatype
/// is derived automatically based on `recv_buf`'s underlying \c value_type.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::recv_buf() is given.
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return If no message is available return \c std::nullopt, else return a \c std::optional wrapping an \ref
/// kamping::MPIResult. If the result object is empty, i.e. there are no owning out parameters passed to `try_recv` (see
/// \ref docs/parameter_handling.md), returns a \c bool indicating success instead of an \c std::optional.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>
/// \include{doc} docs/resize_policy.dox
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::try_recv(Args... args) const {
    // Check parameters
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, tag, source, status, recv_type)
    );

    //  Get the recv buffer
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<recv_value_type_tparam>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        !std::is_same_v<recv_value_type, internal::unused_tparam>,
        "No recv_buf parameter provided and no receive value given as template parameter. One of these is required."
    );

    auto recv_type = internal::determine_mpi_recv_datatype<recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool recv_type_is_in_param = !internal::has_to_be_computed<decltype(recv_type)>;

    // Get the source parameter. If the parameter is not given, use MPI_ANY_SOURCE.
    using default_source_buf_type = decltype(kamping::source(rank::any));
    auto&& source_param =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );

    // Get the tag parameter. If the parameter is not given, use MPI_ANY_TAG.
    using default_tag_buf_type = decltype(kamping::tag(tags::any));
    auto&& tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>({}, args...);
    constexpr auto tag_type = std::remove_reference_t<decltype(tag_param)>::tag_type;
    if constexpr (tag_type == internal::TagType::value) {
        int tag = tag_param.tag();
        KASSERT(
            Environment<>::is_valid_tag(tag),
            "invalid tag " << tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
        );
    }
    // Get the status parameter.
    using default_status_param_type = decltype(kamping::status(kamping::ignore<>));
    auto status_param =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        )
            .construct_buffer_or_rebind();
    KASSERT(internal::is_valid_rank_in_comm(source_param, *this, /*allow_null=*/true, /*allow_any=*/true));
    int source = source_param.rank_signed();
    int tag    = tag_param.tag();

    // Use a matched probe to check if a message with the given source and tag is available for receiving.
    int         msg_avail;
    MPI_Message message;

    Status               status;
    [[maybe_unused]] int err = MPI_Improbe(source, tag, _comm, &msg_avail, &message, &status.native());
    this->mpi_error_hook(err, "MPI_Improbe");

    auto construct_result = [&] {
        return internal::make_mpi_result<std::tuple<Args...>>(
            std::move(recv_buf),
            std::move(status_param),
            std::move(recv_type)
        );
    };
    using result_type = decltype(construct_result());
    // If a message is available, receive it using a matched receive.
    if (msg_avail) {
        size_t const count = status.count(recv_type.get_single_element());

        // Ensure that we do not touch the recv buffer if MPI_PROC_NULL is passed, because this is what the standard
        // guarantees.
        if constexpr (std::remove_reference_t<decltype(source_param)>::rank_type != internal::RankType::null) {
            recv_buf.resize_if_requested([&] { return count; });
            KASSERT(
                // If the recv type is user provided, kamping cannot make any assumptions about the required size of the
                // recv buffer.
                recv_type_is_in_param || recv_buf.size() >= count,
                "Recv buffer is not large enough to hold all received elements.",
                assert::light
            );
        }

        // Use a matched receive to receive exactly the message we probed. This ensures this method is thread-safe.
        err = MPI_Mrecv(
            recv_buf.data(),                                   // buf
            asserting_cast<int>(count),                        // count
            recv_type.get_single_element(),                    // datatype
            &message,                                          // message
            internal::status_param_to_native_ptr(status_param) // status
        );
        this->mpi_error_hook(err, "MPI_Mrecv");

        // Build the result object from the parameters and return.
        if constexpr (is_result_empty_v<result_type>) {
            return true;
        } else {
            return std::optional{construct_result()};
        }
    } else {
        // There was no message to receive, thus return false/std::nullopt.
        if constexpr (is_result_empty_v<result_type>) {
            return false;
        } else {
            return std::optional<result_type>{};
        }
    }
}
/// @}
