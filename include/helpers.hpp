#pragma once

#include <limits>
#include <type_traits>

///
/// @brief Checks if value can be safely casted into type To, that is, it lies in the range [min(To), max(To)].
/// 
/// @tparam To Type to be casted to.
/// @tparam From Type to be casted from, will be auto inferred.
/// @param value Value you want to cast.
/// @return \c true if value can be safely casted into type To, that is, value is in To's range.
/// @return \c false otherwise.
/// @see throwing_cast
/// @see asserting_cast
///
template <class To, class From>
constexpr bool in_range(From value) noexcept {
    static_assert(std::is_integral_v<From>, "From has to be an integral type.");
    static_assert(std::is_integral_v<To>, "To has to be an integral type.");

    static_assert(!std::is_unsigned_v<From> || std::numeric_limits<From>::min() == 0);
    static_assert(!std::is_unsigned_v<To> || std::numeric_limits<To>::min() == 0);

    static_assert(std::numeric_limits<From>::digits <= 64);
    static_assert(std::numeric_limits<To>::digits <= 64);

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
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return constexpr To The casted value.
/// @see in_range
/// @see throwing_cast
///
template <class To, class From>
constexpr To asserting_cast(From value) noexcept {
    assert(in_range<To>(value));
    return static_cast<To>(value);
}

///
/// @brief Casts a value to the type To. If the value is outside To's range, throws an exception.
/// 
/// @tparam To Type to cast to.
/// @tparam From Type to cast from, will be auto inferred.
/// @param value Value you want to cast.
/// @return constexpr To Casted value.
/// @see in_range
/// @see asserting_cast
///
template <class To, class From>
constexpr To throwing_cast(From value) {
    if (!in_range<To>(value)) {
        throw std::range_error("string(value) is not not representable the target type");
    } else {
        return static_cast<To>(value);
    }
}

