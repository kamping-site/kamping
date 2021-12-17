/// @file
/// @brief Macros for optional runtime checks

#pragma once

#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#define KAMPING_SOURCE_LOCATION                 \
    kamping::assert::internal::SourceLocation { \
        __FILE__, __LINE__, __func__            \
    }

#define KAMPING_ASSERT_IMPL(type, expression, message, level)                                                        \
    do {                                                                                                             \
        if constexpr (kamping::assert::internal::assertion_enabled(level)) {                                         \
            if (!kamping::assert::internal::evaluate_assertion(                                                      \
                    type,                                                                                            \
                    kamping::assert::internal::finalize_expr(kamping::assert::internal::Decomposer{} <= expression), \
                    KAMPING_SOURCE_LOCATION, #expression)) {                                                         \
                kamping::assert::Logger<std::ostream&>(std::cerr) << message << "\n";                                \
                std::abort();                                                                                        \
            }                                                                                                        \
        }                                                                                                            \
    } while (false)

#define KASSERT(expression, message, level) KAMPING_ASSERT_IMPL("ASSERTION", expression, message, level)

#ifdef KAMPING_EXCEPTION_MODE
    #define KTHROW(expression, message, assertion_type)                                                                \
        do {                                                                                                           \
            if (!(expression)) {                                                                                       \
                throw assertion_type(                                                                                  \
                    #expression, (kamping::assert::internal::RrefOStringstreamLogger{std::ostringstream{}} << message) \
                                     .stream()                                                                         \
                                     .str());                                                                          \
            }                                                                                                          \
        } while (false)
#else
    #define KTHROW(expression, message, assertion_type) \
        KAMPING_ASSERT_IMPL(#assertion_type, expression, message, kamping::assert::normal)
#endif

namespace kamping::assert {
namespace internal {
/// @internal

/// @brief Sets \c value to \c false for non-streamable types.
template <typename, typename, typename = void>
struct IsStreamableTypeImpl : std::false_type {};

/// @brief Sets \c value to \c true for streamable types.
/// @tparam StreamT An output stream overloading the \c << operator.
/// @tparam ValueT A value type that may or may not be used with \c StreamT::operator<<.
template <typename StreamT, typename ValueT>
struct IsStreamableTypeImpl<StreamT, ValueT, std::void_t<decltype(std::declval<StreamT&>() << std::declval<ValueT>())>>
    : std::true_type {};

/// @brief Determines whether a value of type \c ValueT can be streamed into an output stream of type \c StreamT.
/// @tparam StreamT An output stream overloading the \c << operator.
/// @tparam ValueT A value type that may or may not be used with \c StreamT::operator<<.
template <typename StreamT, typename ValueT>
constexpr bool IsStreamableType = IsStreamableTypeImpl<StreamT, ValueT>::value;

/// @endinternal
} // namespace internal

/// @brief The default exception type used together with \c KTHROW. Reports the erroneous expression together with a
/// custom error message.
class DefaultException : public std::exception {
public:
    /// @brief Constructs the exception based on the erroneous expression and a custom error message.
    /// @param expression The stringified expression that caused this exception to be thrown.
    /// @param message A custom error message.
    explicit DefaultException(std::string const& expression, std::string const& message)
        : _what(build_what(expression, message)) {}

    /// @brief Prints a description of this exception.
    /// @return A description of this exception.
    [[nodiscard]] char const* what() const noexcept final {
        return _what.c_str();
    }

private:
    /// @brief Builds the description of this exception.
    /// @return The description of this exception.
    static std::string build_what(std::string const& expression, std::string const& message) {
        using namespace std::string_literals;
        return "FAILED ASSERTION:"s + "\n\t"s + expression + "\n" + message + "\n";
    }

    /// @brief The description of this exception.
    std::string _what;
};

/// @brief Simple wrapper for output streams that is used to stringify values in assertions and exceptions.
///
/// To enable stringification for custom types, overload the \c << operator of this class.
/// The library overloads this operator for the following STL types:
///
/// * \c std::vector<T>
/// * \c std::pair<K, V>
///
/// \tparam StreamT The underlying streaming object (e.g., \c std::ostream or \c std::ostringstream).
template <typename StreamT>
class Logger {
public:
    /// @brief Construct the object with an underlying streaming object.
    /// @param out The underlying streaming object.
    explicit Logger(StreamT&& out) : _out(std::forward<StreamT>(out)) {}

    /// @brief Forward all values for which \c StreamT::operator<< is defined to the underlying streaming object.
    /// @param value Value to be stringified.
    /// @tparam ValueT Type of the value to be stringified.
    template <typename ValueT, std::enable_if_t<internal::IsStreamableTypeImpl<std::ostream, ValueT>::value, int> = 0>
    Logger<StreamT>& operator<<(ValueT&& value) {
        _out << std::forward<ValueT>(value);
        return *this;
    }

    /// @brief Get the underlying streaming object.
    /// @return The underlying streaming object.
    StreamT&& stream() {
        return std::forward<StreamT>(_out);
    }

private:
    /// @brief The underlying streaming object.
    StreamT&& _out;
};

namespace internal {
/// @internal

/// @brief Stringify a value using the given assertion logger. If the value cannot be streamed into the logger, print
/// \c <?> instead.
/// @tparam StreamT The underlying streaming object of the assertion logger.
/// @tparam ValueT The type of the value to be stringified.
/// @param out The assertion logger.
/// @param value The value to be stringified.
/// @return The stringification of \c value, or \c <?> if the value cannot be stringified by the logger.
template <typename StreamT, typename ValueT>
void stringify_value(Logger<StreamT>& out, const ValueT& value) {
    if constexpr (IsStreamableType<Logger<StreamT>, ValueT>) {
        out << value;
    } else {
        out << "<?>";
    }
}

/// @brief Logger writing all output to a \c std::ostream. This specialization is used to generate the KASSERT error
/// messages.
using OStreamLogger = Logger<std::ostream&>;

/// @brief Logger writing all output to a rvalue \c std::ostringstream. This specialization is used to generate the
/// custom error message for KTHROW exceptions.
using RrefOStringstreamLogger = Logger<std::ostringstream&&>;

/// @endinternal
} // namespace internal

/// @brief Stringification of \c std::vector<T> in assertions.
///
/// Outputs a \c std::vector<T> in the following format, where \code{element i} are the stringified elements of the
/// vector: [element 1, element 2, ...]
///
/// \tparam StreamT The underlying output stream of the Logger.
/// \tparam ValueT The type of the elements contained in the vector.
/// \tparam AllocatorT The allocator of the vector.
/// \param logger The assertion logger.
/// \param container The vector to be stringified.
/// \return The stringified vector as described above.
template <typename StreamT, typename ValueT, typename AllocatorT>
Logger<StreamT>& operator<<(Logger<StreamT>& logger, std::vector<ValueT, AllocatorT> const& container) {
    logger << "[";
    bool first = true;
    for (auto const& element: container) {
        if (!first) {
            logger << ", ";
        }
        logger << element;
        first = false;
    }
    return logger << "]";
}

/// @brief Stringification of \code{std::pair<K, V>} in assertions.
///
/// Outputs a \code{std::pair<K, V>} in the following format, where \c first and \c second are the stringified
/// components of the pair: (first, second)
///
/// \tparam StreamT The underlying output stream of the Logger.
/// \tparam Key Type of the first component of the pair.
/// \tparam Value Type of the second component of the pair.
/// \param logger The assertion logger.
/// \param pair The pair to be stringified.
/// \return The stringification of the pair as described above.
template <typename StreamT, typename Key, typename Value>
Logger<StreamT>& operator<<(Logger<StreamT>& logger, std::pair<Key, Value> const& pair) {
    return logger << "(" << pair.first << ", " << pair.second << ")";
}

namespace internal {
/// @internal

class Expr {
public:
    virtual ~Expr() = default;

    [[nodiscard]] virtual bool result() const = 0;

    virtual void stringify(OStreamLogger& out) const = 0;

    friend OStreamLogger& operator<<(OStreamLogger& out, Expr const& expr) {
        expr.stringify(out);
        return out;
    }
};

template <typename LhsT, typename RhsT>
class BinaryExpr : public Expr {
public:
    BinaryExpr(bool const result, LhsT const& lhs, const std::string_view op, RhsT const& rhs)
        : _result(result),
          _lhs(lhs),
          _op(op),
          _rhs(rhs) {}

    [[nodiscard]] bool result() const final {
        return _result;
    }

    void stringify(OStreamLogger& out) const final {
        stringify_value(out, _lhs);
        out << " " << _op << " ";
        stringify_value(out, _rhs);
    }

#define KAMPING_ASSERT_OP(op)                                                                                     \
    template <typename RhsPrimeT>                                                                                 \
    friend BinaryExpr<BinaryExpr<LhsT, RhsT>, RhsPrimeT> operator op(                                             \
        BinaryExpr<LhsT, RhsT>&& lhs, RhsPrimeT const& rhs_prime) {                                               \
        using namespace std::string_view_literals;                                                                \
        return BinaryExpr<BinaryExpr<LhsT, RhsT>, RhsPrimeT>(lhs.result() op rhs_prime, lhs, #op##sv, rhs_prime); \
    }

    KAMPING_ASSERT_OP(&&)
    KAMPING_ASSERT_OP(||)

#undef KAMPING_ASSERT_OP

private:
    bool             _result;
    LhsT const&      _lhs;
    std::string_view _op;
    RhsT const&      _rhs;
};

template <typename LhsT>
class UnaryExpr : public Expr {
    explicit UnaryExpr(const LhsT& lhs) : _lhs(lhs) {}

    [[nodiscard]] bool result() const final {
        return static_cast<bool>(_lhs);
    }

    void stringify(OStreamLogger& out) const final {
        stringify_value(out, _lhs);
    }

private:
    const LhsT& _lhs;
};

template <typename LhsT>
class LhsExpr {
public:
    explicit LhsExpr(LhsT const& lhs) : _lhs(lhs) {}

    UnaryExpr<LhsT> make_unary() {
        static_assert(std::is_convertible_v<LhsT, bool>, "expression must be convertible to bool");
        return {_lhs};
    }

#define KAMPING_ASSERT_OP(op)                                                   \
    template <typename RhsT>                                                    \
    friend BinaryExpr<LhsT, RhsT> operator op(LhsExpr&& lhs, RhsT const& rhs) { \
        using namespace std::string_view_literals;                              \
        return {lhs._lhs op rhs, lhs._lhs, #op##sv, rhs};                       \
    }

    KAMPING_ASSERT_OP(==)
    KAMPING_ASSERT_OP(!=)
    KAMPING_ASSERT_OP(&&)
    KAMPING_ASSERT_OP(||)
    KAMPING_ASSERT_OP(<)
    KAMPING_ASSERT_OP(<=)
    KAMPING_ASSERT_OP(>)
    KAMPING_ASSERT_OP(>=)
    KAMPING_ASSERT_OP(&)
    KAMPING_ASSERT_OP(|)
    KAMPING_ASSERT_OP(^)

#undef KAMPING_ASSERT_OP

private:
    const LhsT& _lhs;
};

struct Decomposer {
    template <typename LhsT>
    friend LhsExpr<LhsT> operator<=(Decomposer&&, LhsT const& lhs) {
        return LhsExpr<LhsT>(lhs);
    }
};

struct SourceLocation {
    char const* file;
    unsigned    row;
    char const* function;
};

constexpr bool assertion_enabled(int level) {
    return level <= KAMPING_ASSERTION_LEVEL;
}

template <typename ExprT>
Expr&& finalize_expr(ExprT&& expr) {
    if constexpr (std::is_base_of_v<Expr, std::remove_reference_t<std::remove_const_t<ExprT>>>) {
        return std::forward<ExprT>(expr);
    } else {
        return expr.make_unary();
    }
}

bool evaluate_assertion(char const* type, Expr&& expr, const SourceLocation& where, char const* expr_str) {
    if (!expr.result()) {
        OStreamLogger{std::cerr} << where.file << ": In function '" << where.function << "':\n"
                                 << where.file << ":" << where.row << ": FAILED " << type << "\n"
                                 << "\t" << expr_str << "\n"
                                 << "with expansion:\n"
                                 << "\t" << expr << "\n";
    }
    return expr.result();
}

/// @endinternal
} // namespace internal

/// @name Predefined assertion levels
/// Assertion levels that can be used with the KASSERT macro.
/// @{
/// @brief Assertion level for lightweight assertions.
constexpr int light = 1;
/// @brief Default assertion level. This level is used if no assertion level is specified.
constexpr int normal = 2;
/// @brief Assertion level for heavyweight assertions.
constexpr int heavy = 3;
/// @}
} // namespace kamping::assert
