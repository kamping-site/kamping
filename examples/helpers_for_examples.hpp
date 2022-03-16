// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include "kamping/communicator.hpp"

namespace kamping {
template <typename T>
void print_result(std::vector<T> const& result, Communicator comm) {
    if (comm.rank() == 0) {
        for (auto const& elem: result) {
            std::cout << elem << "\n";
        }
        std::cout << std::flush;
    }
}
} // namespace kamping
