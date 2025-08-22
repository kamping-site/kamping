#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/named_parameter_check.hpp"

template <typename... Args>
inline std::size_t pack_size(Args... args) {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(send_buf, send_count, send_type)
    );
    auto send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();

    using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

    auto send_type = internal::determine_mpi_send_datatype<send_value_type>(args...);

    using default_send_count_type = decltype(kamping::send_count_out());
    auto send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
            {},
            args...
        )
            .construct_buffer_or_rebind();
    if constexpr (has_to_be_computed<decltype(send_count)>) {
        send_count.underlying() = asserting_cast<int>(send_buf.size());
    }
    int pack_size = 0;
    MPI_Pack_size(
        send_count.get_single_element(), // incount
        send_type.get_single_element(),  // datatype
        MPI_COMM_WORLD,                  // TODO
        &pack_size                       // size
    );
    return asserting_cast<std::size_t>(pack_size);
}
