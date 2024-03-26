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

/// @brief dummy default argument
struct DefaultArgument {
    DefaultArgument(int value, std::string message) : _value(value), _message(message) {}
    int         _value;
    std::string _message;
};

} // namespace kamping

int main() {
    using namespace kamping;
    using SendBuffer_Arg = Argument<internal::ParameterType::send_buf>;
    using SendCounts_Arg = Argument<internal::ParameterType::send_counts>;

    SendBuffer_Arg const arg_id_1(1);
    SendBuffer_Arg const arg_id_2(2);
    SendCounts_Arg const arg_id_3(3);
    SendCounts_Arg const arg_id_4(4);

    {
        // first argument within parameter list with ParameterType send_buf is selected
        auto const& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_buf>(arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    {
        // first argument within parameter list with ParameterType send_counts is selected
        auto const& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_counts>(arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    {
        // first argument within parameter list with ParameterType send_buf is selected
        auto const& selected_arg =
            internal::select_parameter_type<internal::ParameterType::send_buf>(arg_id_2, arg_id_1, arg_id_3, arg_id_4);
        std::cout << "Id of selected Argument: " << selected_arg.id << std::endl;
    }
    {
        // We can provide default arguments which are only constructed if the parameter is not given
        // arguments to the default parameter are passed as a tuple
        auto&& selected_arg =
            internal::select_parameter_type_or_default<internal::ParameterType::root, DefaultArgument>(
                std::tuple(42, "KaMPIng"),
                arg_id_2,
                arg_id_1,
                arg_id_3,
                arg_id_4
            );
        std::cout << "parameters of default argument: " << selected_arg._value << " " << selected_arg._message
                  << std::endl;
    }
    return 0;
}
