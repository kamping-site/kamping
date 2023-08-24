// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
/// This file contains functionality that is related to measurement aggregation.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace kamping::measurements {
///@brief Either a scalar or vector of type \c T.
///@tparam T Type.
template <typename T>
using ScalarOrContainer = std::variant<T, std::vector<T>>;

/// @brief Enum to specify how time measurements with same key shall be aggregated locally.
enum class LocalAggregationMode {
    accumulate, ///< Tag used to indicate that data associated with identical keys will be accumulated into a scalar.
    append      ///< Tag used to indicate that data with identical keys will not be accumulated and stored in a list.
};

/// @brief Enum to specify how time durations with same key shall be aggregated across the participating ranks.
enum class GlobalAggregationMode {
    min,   ///< The minimum of the measurement data on the participating ranks will be computed.
    max,   ///< The maximum of the measurement data on the participating ranks will be computed.
    sum,   ///< The sum of the measurement data on the participating ranks will be computed.
    gather ///< The measurement data on the participating ranks will be collected in a container.
};

/// @brief Returns name of given GlobalAggregationMode.
/// @param mode Given mode for which a name as a string is requested.
/// @return Name of mode as a string.
inline std::string get_string(GlobalAggregationMode mode) {
    switch (mode) {
        case GlobalAggregationMode::min:
            return "min";
        case GlobalAggregationMode::max:
            return "max";
        case GlobalAggregationMode::sum:
            return "sum";
        case GlobalAggregationMode::gather:
            return "gather";
    }
    return "No name string is specified for given mode.";
}

} // namespace kamping::measurements
