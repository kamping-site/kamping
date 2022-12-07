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

#include <utility>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::recv(Args... args) const {
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(recv_buf),
        KAMPING_OPTIONAL_PARAMETERS(tag, source, recv_counts, status)
    );

    auto& recv_buf_param  = internal::select_parameter_type<internal::ParameterType::recv_buf>(args...);
    auto  recv_buf        = recv_buf_param.get();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf_param)>::value_type;

    using default_source_buf_type = decltype(kamping::source(rank::any));

    auto&& source =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );

    using default_tag_buf_type = decltype(kamping::tag(tags::any));

    auto&& tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>({}, args...);
    int tag = tag_param.tag();

    constexpr auto tag_type = std::remove_reference_t<decltype(tag_param)>::tag_type;
    if constexpr (tag_type == internal::TagType::value) {
        KASSERT(is_valid_tag(tag), "invalid tag " << tag << ", maximum allowed tag is " << tag_upper_bound());
    }

    using default_status_param_type = decltype(kamping::status(kamping::ignore<>));

    auto&& status =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        );

    // Get the optional recv_count parameter. If the parameter is not given,
    // allocate a new container.
    using default_recv_count_type = decltype(kamping::recv_counts_out(NewContainer<int>{}));
    auto&& recv_count_param =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );

    constexpr bool recv_count_is_output_parameter = internal::has_to_be_computed<decltype(recv_count_param)>;
    if constexpr (recv_count_is_output_parameter) {
        auto probe_status        = this->probe(source, tag_param, status_out()).status();
        *recv_count_param.data() = probe_status.template count<recv_value_type>();
    }

    constexpr auto rank_type = std::remove_reference_t<decltype(source)>::rank_type;
    if constexpr (rank_type == internal::RankType::value) {
        KASSERT(this->is_valid_rank(source.rank_signed()), "Invalid receiver rank.");
    }
    recv_buf.resize(recv_count_param.get_single_element());

    [[maybe_unused]] int err = MPI_Recv(
        recv_buf.data(),                                            // buf
        asserting_cast<int>(recv_count_param.get_single_element()), // count
        mpi_datatype<recv_value_type>(),                            // dataype
        source.rank_signed(),                                       // source
        tag,                                                        // tag
        this->mpi_communicator(),                                   // comm
        status.native_ptr()                                         // status
    );
    THROW_IF_MPI_ERROR(err, MPI_Recv);

    return make_mpi_result(std::move(recv_buf), std::move(status));
}

// template <typename T, typename... Args>
// auto recv_single(Args... args) const {}
