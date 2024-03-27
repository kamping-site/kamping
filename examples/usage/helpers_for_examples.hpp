// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include "kamping/communicator.hpp"

namespace kamping {
/// @brief Print all elements in a container, prefixed with the rank of the current PE.
/// @tparam ContainerType Type of the communicator's default container.
/// @tparam Plugins Types of the communicator's plugins.
/// @tparam T Type of the elements contained in the container.
/// @param result The container whose elements are printed.
/// @param comm KaMPIng communicator to get the rank of the PE.
template <
    template <typename...>
    typename ContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins,
    typename T>
void print_result(ContainerType<T> const& result, Communicator<ContainerType, Plugins...> const& comm) {
    for (auto const& elem: result) {
        std::cout << "[PE " << comm.rank() << "] " << elem << "\n";
    }
    std::cout << std::flush;
}

/// @brief Print the given element, prefixed with the rank of the current PE.
/// @tparam T Type of the element.
/// @tparam ContainerType Type of the communicator's default container.
/// @tparam Plugins Types of the communicator's plugins.
/// @param result The elements to be printed. Streamed to std::cout.
/// @param comm KaMPIng communicator to get the rank of the PE.
template <
    template <typename...>
    typename ContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins,
    typename T>
void print_result(T const& result, Communicator<ContainerType, Plugins...> const& comm) {
    std::cout << "[PE " << comm.rank() << "] " << result << std::endl;
}

/// @brief Print all elements in a container only on the root PE.
/// @tparam T Type of the elements contained in the container.
/// @param result The container whose elements are printed on the root PE.
/// @param comm KaMPIng communicator to determine which PE is the root PE.
template <template <typename...> typename ContainerType, typename T, typename Comm>
void print_result_on_root(ContainerType<T> const& result, Comm const& comm) {
    if (comm.is_root()) {
        print_result(result, comm);
    }
}

/// @brief Print the given string only on the root PE.
/// @tparam Communicator Type of communicator (has to be a KaMPIng communicator).
/// @param str The string to be printed.
/// @param comm KaMPIng communicator to determine which PE is the root PE.
template <typename Communicator>
void print_on_root(std::string const& str, Communicator const& comm) {
    if (comm.is_root()) {
        print_result(str, comm);
    }
}
} // namespace kamping
