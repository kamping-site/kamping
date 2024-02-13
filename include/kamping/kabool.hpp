// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
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
namespace kamping {
/// @brief Wrapper around bool to allow handling containers of boolean values
class kabool {
public:
    /// @brief default constructor for a \c kabool with value \c false
    constexpr kabool() noexcept : _value() {}
    /// @brief constructor to construct a \c kabool out of a \c bool
    constexpr kabool(bool value) noexcept : _value(value) {}

    /// @brief implicit cast of \c kabool to \c bool
    inline constexpr operator bool() const noexcept {
        return _value;
    }

private:
    bool _value; /// < the wrapped boolean value
};
} // namespace kamping
