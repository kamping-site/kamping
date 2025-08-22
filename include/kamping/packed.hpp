// This file is part of KaMPIng.
//
// Copyright 2021-2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <cstddef>

namespace kamping {
/// @brief Wrapper around \c std::byte to allow handling containers of \c MPI_PACKED values
class packed {
public:
    /// @brief default constructor for a default initialized \c packed.
    constexpr packed() noexcept : _value() {}
    /// @brief constructor to construct a \c packed out of \c std::byte
    constexpr packed(std::byte value) noexcept : _value(value) {}

    /// @brief implicit cast of \c packed to \c std::byte
    inline constexpr operator std::byte() const noexcept {
        return _value;
    }

private:
    std::byte _value; /// < the wrapped byte
};
} // namespace kamping
