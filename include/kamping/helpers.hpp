/// @file
/// @brief Helper functions that make casts safer.

#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

/// @addtogroup kamping_utility
/// @{
///
/// @brief Checks if value can be safely casted into type To, that is, it lies in the range [min(To), max(To)].
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
        std::is_signed_v<From> || std::numeric_limits<From>::min() == 0, "The type From has to include the number 0.");
    static_assert(
        std::is_signed_v<To> || std::numeric_limits<To>::min() == 0, "The type To has to include the number 0.");

    // Check if we can safely cast To and From into (u)intmax_t.
    if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        static_assert(
            std::numeric_limits<From>::digits <= std::numeric_limits<intmax_t>::digits,
            "From has more bits than intmax_t.");
        static_assert(
            std::numeric_limits<To>::digits <= std::numeric_limits<intmax_t>::digits,
            "To has more bits than intmax_t.");
    } else {
        static_assert(
            std::numeric_limits<From>::digits <= std::numeric_limits<uintmax_t>::digits,
            "From has more bits than uintmax_t.");
        static_assert(
            std::numeric_limits<To>::digits <= std::numeric_limits<uintmax_t>::digits,
            "To has more bits than uintmax_t.");
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
/// @brief Casts a value to the type To. If the value is outside To's range, throws an assertion.
///
/// Alternatively, exceptions can be used instead of assertions by using \ref trowing_cast.
///
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return constexpr To The casted value.
///
template <class To, class From>
constexpr To asserting_cast(From value) noexcept {
    assert(in_range<To>(value));
    return static_cast<To>(value);
}

///
/// @brief Casts a value to the type To. If the value is outside To's range, throws an exception.
///
/// Alternatively, assertions can be used instead of exceptions by using \ref asserting_cast.
///
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return constexpr To Casted value.
///
template <class To, class From>
constexpr To throwing_cast(From value) {
    if (!in_range<To>(value)) {
        throw std::range_error(std::to_string(value) + " is not not representable the target type.");
    } else {
        return static_cast<To>(value);
    }
}

///@}
