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

/// @file
/// @brief The header file for the second compilation unit of a test that checks if compiling and running works
/// correctly when linking two compilation units that both use KaMPIng

#include <vector>

/// @brief A class that provides a gather function for single ints
class Gatherer {
public:
    /// @brief Collective operation: Gather the ints provided on each rank on the root.
    ///
    /// @param data the int provided on this rank.
    /// @return The gathered data on the root. An empty vector on all other ranks.
    std::vector<int> gather(int data);
};
