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
