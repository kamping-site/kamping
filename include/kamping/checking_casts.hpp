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

/// @file
/// @brief Helper functions that make casts safer.

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <kassert/kassert.hpp>

#include "kamping/noexcept.hpp"

namespace kamping {

/// @addtogroup kamping_utility
/// @{

/// @brief Checks if an integer value can be safely casted into an integer type To, that is, it lies in the range
/// [min(To), max(To)].
///
/// This function works only for integer types which have at most std::numeric_limits<intmax_t>::digits (To and From are
/// signed) or std::numeric_limits<intmax_t>::digits (else) bits. This function includes checks for these two
/// assumptions using static_assert()s.
///
/// @tparam To Type to be casted to.
/// @tparam From Type to be casted from, will be auto inferred.
/// @param value Value you want to cast.
/// @return \c true if value can be safely casted into type To, that is, value is in To's range.
/// @return \c false otherwise.
///
template <class To, class From>
constexpr bool in_range(From value) noexcept {
    static_assert(std::is_integral_v<From>, "From has to be an integral type.");
    static_assert(std::is_integral_v<To>, "To has to be an integral type.");

    // Check that the 0 is included in From and To. 0 is always included in signed types.
    static_assert(
        std::is_signed_v<From> || std::numeric_limits<From>::min() == 0,
        "The type From has to include the number 0."
    );
    static_assert(
        std::is_signed_v<To> || std::numeric_limits<To>::min() == 0,
        "The type To has to include the number 0."
    );

    // Check if we can safely cast To and From into (u)intmax_t.
    if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        static_assert(
            std::numeric_limits<From>::digits <= std::numeric_limits<intmax_t>::digits,
            "From has more bits than intmax_t."
        );
        static_assert(
            std::numeric_limits<To>::digits <= std::numeric_limits<intmax_t>::digits,
            "To has more bits than intmax_t."
        );
    } else {
        static_assert(
            std::numeric_limits<From>::digits <= std::numeric_limits<uintmax_t>::digits,
            "From has more bits than uintmax_t."
        );
        static_assert(
            std::numeric_limits<To>::digits <= std::numeric_limits<uintmax_t>::digits,
            "To has more bits than uintmax_t."
        );
    }

    // Check if the parameters value is inside To's range.
    if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        return static_cast<intmax_t>(value) >= static_cast<intmax_t>(std::numeric_limits<To>::min())
               && static_cast<intmax_t>(value) <= static_cast<intmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
        if (value < 0) {
            return false;
        } else {
            return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
        }
    } else if constexpr (std::is_unsigned_v<From> && std::is_signed_v<To>) {
        return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
    }
}

///
/// @brief Casts an integer value to the integer type To. If the value is outside To's range, throws an assertion.
///
/// Alternatively, exceptions can be used instead of assertions by using \ref throwing_cast().
///
/// This function works only for integer types which have at most std::numeric_limits<intmax_t>::digits (To and From are
/// signed) or std::numeric_limits<intmax_t>::digits (else) bits. These two assumptions are checked by in_range() using
/// static_assert()s.
///
///
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return Casted value.
///
template <class To, class From>
constexpr To asserting_cast(From value) KAMPING_NOEXCEPT {
    KASSERT(in_range<To>(value));
    return static_cast<To>(value);
}

///
/// @brief Casts an integer value to the integer type To. If the value is outside To's range, throws an exception.
///
/// Alternatively, assertions can be used instead of exceptions by using \ref asserting_cast().
///
/// This function works only for integer types which have at most std::numeric_limits<intmax_t>::digits (To and From are
/// signed) or std::numeric_limits<intmax_t>::digits (else) bits. These two assumptions are checked by in_range() using
/// static_assert()s.
///
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return Casted value.
///
template <class To, class From>
constexpr To throwing_cast(From value) {
    THROWING_KASSERT_SPECIFIED(
        in_range<To>(value),
        value << " is not representable by the target type.",
        std::range_error
    );
    return static_cast<To>(value);
}

/// @}

} // namespace kamping
