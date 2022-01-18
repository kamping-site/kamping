// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>

#include "kamping/named_parameter_selection.hpp"

namespace kamping {
// Mock object to simulate a kamping MPI Object.
template <internal::ParameterType parameter_type_>
struct Argument {
    static constexpr internal::ParameterType parameter_type = parameter_type_;

    Argument(std::size_t id_) : id{id_} {}
    std::size_t id;
};


} // namespace kamping

int main() {
    using namespace kamping;
    using SendBuffer_Arg = Argument<internal::ParameterType::send_buf>;
    using SendCounts_Arg = Argument<internal::ParameterType::send_counts>;

    const SendBuffer_Arg arg_id_1(1);
    const SendBuffer_Arg arg_id_2(2);
    const SendCounts_Arg arg_id_3(3);
    const SendCounts_Arg arg_id_4(4);

    {
        // first argument within parameter list with ParameterType send_buf is selected
        const auto& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_buf>(arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    {
        // first argument within parameter list with ParameterType send_counts is selected
        const auto& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_counts>(arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    {
        // first argument within parameter list with ParameterType send_buf is selected
        const auto& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_buf>(arg_id_2, arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    return 0;
}
