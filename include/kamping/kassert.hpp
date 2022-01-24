// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Macros for asserting runtime checks.

#pragma once

#include <iostream>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Assertion macro for the KaMPI.ng library. Accepts between one and three parameters.
///
/// If the assertion fails, the KASSERT() macro prints an expansion of the expression similar to Catch2. For instance,
/// in a call
///     KASSERT(rhs == lhs)
/// KASSERT() also prints the values of \c rhs and \c lhs. However, this expression expansion is limited and only works
/// for expressions that do not contain parentheses, but are implicitly left-associative. This is due to its
/// implementation:
///     KASSERT(rhs == lhs)
/// is replaced by
///     Decomposer{} <= rhs == lhs
/// which is interpreted by the compiler as
///     ((Decomposer{} <= rhs) == lhs)
/// where the first <= relation is overloaded to return a proxy object which in turn overloads other operators. If the
/// expression is not implicitly left-associative or contains parentheses, this does not work:
///     KASSERT(rhs1 == lhs1 && rhs2 == lhs2)
/// is replaced by (with implicit parentheses)
///     ((Decomposer{} <= rhs1) == lhs1) && (rhs2 == lhs2))
/// Thus, the left hand side of \c && can only be expanded to the *result* of \code{rhs2 == lhs2}.
/// This limitation only affects the error message, not the interpretation of the expression itself.
///
/// @param expression The assertion expression (mandatory argument).
/// @param message Error message that is printed in addition to the decomposed expression (optional argument).
/// @param level The level of the assertion (optional arguments, defaults to \c kamping::assert::normal).
#define KASSERT(...)               \
    KAMPING_KASSERT_VARARG_HELPER( \
        , __VA_ARGS__, KASSERT_3(__VA_ARGS__), KASSERT_2(__VA_ARGS__), KASSERT_1(__VA_ARGS__), ignore)

/// @brief Macro for throwing exceptions inside the KaMPI.ng library. Accepts between one and three parameters.
/// @param expression Expression that causes the exception to be thrown if it evaluates the \c false
/// (mandatory argument).
/// @param message Error message that is added to the exception (optional argument).
/// @param assertion_type Type of the exception that should be thrown
/// (optional argument, defaults to \c kamping::assert::KassertException).
#define KTHROW(...)                \
    KAMPING_KASSERT_VARARG_HELPER( \
        , __VA_ARGS__, KTHROW_3(__VA_ARGS__), KTHROW_2(__VA_ARGS__), KTHROW_1(__VA_ARGS__), ignore)

// To decompose expressions, the KAMPING_KASSERT_HPP_ASSERT_IMPL() produces code such as
//
// Decomposer{} <= a == b       [ with implicit parentheses: ((Decomposer{} <= a) == b) ]
//
// This triggers a warning with -Wparentheses, suggesting to set explicit parentheses, which is impossible in this
// situation. Thus, we use compiler-specific _Pragmas to suppress these warning.
// Note that warning suppression in GCC does not work if the KASSERT() call is passed through >= two macro calls:
//
// #define A(stmt) B(stmt)
// #define B(stmt) stmt;
// A(KASSERT(1 != 1)); -- warning suppression does not work
//
// This is a known limitation of the current implementation.
#if defined(__GNUC__) && !defined(__clang__) // GCC
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_PUSH               _Pragma("GCC diagnostic push")
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_POP                _Pragma("GCC diagnostic pop")
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_IGNORE_PARENTHESES _Pragma("GCC diagnostic ignored \"-Wparentheses\"")
#elif defined(__clang__) // Clang
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_PUSH               _Pragma("clang diagnostic push")
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_POP                _Pragma("clang diagnostic pop")
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_IGNORE_PARENTHESES _Pragma("clang diagnostic ignored \"-Wparentheses\"")
#else // Other compilers -> no supression supported
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_PUSH
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_POP
    #define KAMPING_KASSERT_HPP_DIAGNOSTIC_IGNORE_PARENTHESES
#endif

// This is the actual implementation of the KASSERT() macro.
//
// - Note that expanding the macro into a `do { ... } while(false)` pseudo-loop is a common trick to make a macro
//   "act like a statement". Otherwise, it would have surprising effects if the macro is used inside a `if` branch
//   without braces.
// - If the assertion level is disabled, this should not generate any code (assuming that the compiler removes the
//   dead loop).
// - `evaluate_and_print_assertion` evaluates the assertion and prints an error message if it failed.
// - The call to `std::abort()` is not wrapped in a function to keep the stack trace clean.
#define KAMPING_KASSERT_HPP_KASSERT_IMPL(type, expression, message, level)                                           \
    do {                                                                                                             \
        if constexpr (kamping::assert::internal::assertion_enabled(level)) {                                         \
            KAMPING_KASSERT_HPP_DIAGNOSTIC_PUSH                                                                      \
            KAMPING_KASSERT_HPP_DIAGNOSTIC_IGNORE_PARENTHESES                                                        \
            if (!kamping::assert::internal::evaluate_and_print_assertion(                                            \
                    type,                                                                                            \
                    kamping::assert::internal::finalize_expr(kamping::assert::internal::Decomposer{} <= expression), \
                    KAMPING_KASSERT_HPP_SOURCE_LOCATION, #expression)) {                                             \
                kamping::assert::Logger<std::ostream&>(std::cerr) << message << "\n";                                \
                std::abort();                                                                                        \
            }                                                                                                        \
            KAMPING_KASSERT_HPP_DIAGNOSTIC_POP                                                                       \
        }                                                                                                            \
    } while (false)

// Expands a macro depending on its number of arguments. For instance,
//
// #define FOO(...) KAMPING_KASSERT_VARARG_HELPER(, __VA_ARGS__, IMPL3, IMPL2, IMPL1, dummy)
//
// expands to IMPL3 with 3 arguments, IMPL2 with 2 arguments and IMPL1 with 1 argument.
// To do this, the macro always expands to its 5th argument. Depending on the number of parameters, __VA_ARGS__
// pushes the right implementation to the 5th parameter.
#define KAMPING_KASSERT_VARARG_HELPER(X, Y, Z, W, FUNC, ...) FUNC

// KASSERT() chooses the right implementation depending on its number of arguments.
#define KASSERT_3(expression, message, level) KAMPING_KASSERT_HPP_KASSERT_IMPL("ASSERTION", expression, message, level)
#define KASSERT_2(expression, message)        KASSERT_3(expression, message, kamping::assert::normal)
#define KASSERT_1(expression)                 KASSERT_2(expression, "")

// Implementation of the KTHROW() macro.
// In KAMPING_EXCEPTION_MODE, we throw an exception similar to the implementation of KASSERT(), although expression
// decomposition in exceptions is currently unsupported. Otherwise, the macro delegates to KASSERT().
#ifdef KAMPING_EXCEPTION_MODE
    #define KAMPING_KASSERT_HPP_KTHROW_IMPL(expression, message, assertion_type)                              \
        do {                                                                                                  \
            if constexpr (kamping::assert::internal::assertion_enabled(kamping::assert::exception)) {         \
                if (!(expression)) {                                                                          \
                    throw assertion_type(                                                                     \
                        #expression,                                                                          \
                        (kamping::assert::internal::RrefOStringstreamLogger{std::ostringstream{}} << message) \
                            .stream()                                                                         \
                            .str());                                                                          \
                }                                                                                             \
            }                                                                                                 \
        } while (false)
#else
    #define KAMPING_KASSERT_HPP_KTHROW_IMPL(expression, message, assertion_type) \
        KAMPING_KASSERT_HPP_KASSERT_IMPL(#assertion_type, expression, message, kamping::assert::exception)
#endif

// KTHROW() chooses the right implementation depending on its number of arguments.
#define KTHROW_3(expression, message, assertion_type) \
    KAMPING_KASSERT_HPP_KTHROW_IMPL(expression, message, assertion_type)
#define KTHROW_2(expression, message) KTHROW_3(expression, message, kamping::assert::KassertException)
#define KTHROW_1(expression)          KTHROW_2(expression, "")

// __PRETTY_FUNCTION__ is a compiler extension supported by GCC and clang that prints more information than __func__
#if defined(__GNUC__) || defined(__clang__)
    #define KAMPING_KASSERT_HPP_FUNCTION_NAME __PRETTY_FUNCTION__
#else
    #define KAMPING_KASSERT_HPP_FUNCTION_NAME __func__
#endif

// Represents the static location in the source code.
#define KAMPING_KASSERT_HPP_SOURCE_LOCATION                   \
    kamping::assert::internal::SourceLocation {               \
        __FILE__, __LINE__, KAMPING_KASSERT_HPP_FUNCTION_NAME \
    }

namespace kamping::assert {
/// @name Predefined assertion levels
/// Assertion levels that can be used with the KASSERT macro.
/// @{
/// @brief Assertion level for exceptions if exception mode is disabled.
constexpr int exception = 1;
/// @brief Assertion level for lightweight assertions.
constexpr int light = 2;
/// @brief Default assertion level. This level is used if no assertion level is specified.
constexpr int normal = 3;
/// @brief Assertion level for heavyweight assertions.
constexpr int heavy = 4;
/// @}

namespace internal {
/// @internal

// If partially specialized template is not applicable, set value to false.
template <typename, typename, typename = void>
struct IsStreamableTypeImpl : std::false_type {};

// Partially specialize template if StreamT::operator<<(ValueT) is valid.
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
class KassertException : public std::exception {
public:
    /// @brief Constructs the exception based on the erroneous expression and a custom error message.
    /// @param expression The stringified expression that caused this exception to be thrown.
    /// @param message A custom error message.
    explicit KassertException(std::string const& expression, std::string const& message)
        : _what(_build_what(expression, message)) {}

    /// @brief Prints a description of this exception.
    /// @return A description of this exception.
    [[nodiscard]] char const* what() const noexcept final {
        return _what.c_str();
    }

private:
    /// @brief Builds the description of this exception.
    /// @return The description of this exception.
    static std::string _build_what(std::string const& expression, std::string const& message) {
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
/// @tparam StreamT The underlying streaming object (e.g., \c std::ostream or \c std::ostringstream).
template <typename StreamT>
class Logger {
public:
    /// @brief Construct the object with an underlying streaming object.
    /// @param out The underlying streaming object.
    explicit Logger(StreamT&& out) : _out(std::forward<StreamT>(out)) {}

    /// @brief Forward all values for which \c StreamT::operator<< is defined to the underlying streaming object.
    /// @param value Value to be stringified.
    /// @tparam ValueT Type of the value to be stringified.
    template <typename ValueT, std::enable_if_t<internal::IsStreamableType<std::ostream, ValueT>, int> = 0>
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
void stringify_value(Logger<StreamT>& out, ValueT const& value) {
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
/// components of the pair: (first, second).
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

/// @name Expression expansion
///
/// Implements the decomposition and expansion of expressions. Example usage:
/// @code
/// OStreamLogger(std::cerr) << finalize_expr(Decomposer{} <= <expression>);
/// @endcode
/// \c Decomposer wraps the first part of an expression in a \c LhsExpression, which overloads binary operators and
/// relations to create objects of \c BinaryExpression. \c BinaryExpression overloads \c && and \c || to build chained
/// expressions. If the expression does not contain any operators or relations, it is turned into an \c UnaryExpression
/// by \c finalize_expr if and only if it is implicitly convertible to bool; otherwise, the expression is malformed an a
/// static assertion is triggered.
///
/// @{

/// @brief Interface for decomposed unary and binary expressions.
class Expression {
public:
    /// @brief Virtual destructor since we use virtual functions.
    virtual ~Expression() = default;

    /// @brief Evaluate the assertion wrapped in this Expr.
    /// @return The boolean value that the assertion evalutes to.
    [[nodiscard]] virtual bool result() const = 0;

    /// @brief Write this expression with stringified operands to the given assertion logger.
    /// @param out The assertion logger.
    virtual void stringify(OStreamLogger& out) const = 0;

    /// @brief Writes an expression with stringified operands to the given assertion logger.
    /// @param out The assertion logger.
    /// @param expr The expression to be stringified.
    /// @return The assertion logger.
    friend OStreamLogger& operator<<(OStreamLogger& out, Expression const& expr) {
        expr.stringify(out);
        return out;
    }
};

/// @brief A decomposed binary expression.
/// @tparam LhsT Decomposed type of the left hand side of the expression.
/// @tparam RhsT Decomposed type of the right hand side of the expression.
template <typename LhsT, typename RhsT>
class BinaryExpression : public Expression {
public:
    /// @brief Constructs a decomposed binary expression.
    /// @param result Boolean result of the expression.
    /// @param lhs Decomposed left hand side of the expression.
    /// @param op Stringified operator or relation.
    /// @param rhs Decomposed right hand side of the expression.
    BinaryExpression(bool const result, LhsT const& lhs, std::string_view const op, RhsT const& rhs)
        : _result(result),
          _lhs(lhs),
          _op(op),
          _rhs(rhs) {}

    /// @brief The boolean result of the expression.
    /// @return The boolean result of the expression.
    [[nodiscard]] bool result() const final {
        return _result;
    }

    /// @brief Writes this expression with stringified operands to the given assertion logger.
    /// @param out The assertion logger.
    void stringify(OStreamLogger& out) const final {
        stringify_value(out, _lhs);
        out << " " << _op << " ";
        stringify_value(out, _rhs);
    }

    // Overload operators to return a proxy object that decomposes the rhs of the logical operator
#define KAMPING_ASSERT_OP(op)                                                     \
    template <typename RhsPrimeT>                                                 \
    friend BinaryExpression<BinaryExpression<LhsT, RhsT>, RhsPrimeT> operator op( \
        BinaryExpression<LhsT, RhsT>&& lhs, RhsPrimeT const& rhs_prime) {         \
        using namespace std::string_view_literals;                                \
        return BinaryExpression<BinaryExpression<LhsT, RhsT>, RhsPrimeT>(         \
            lhs.result() op rhs_prime, lhs, #op##sv, rhs_prime);                  \
    }

    KAMPING_ASSERT_OP(&&)
    KAMPING_ASSERT_OP(||)
    KAMPING_ASSERT_OP(&)
    KAMPING_ASSERT_OP(|)
    KAMPING_ASSERT_OP(^)
    KAMPING_ASSERT_OP(==)
    KAMPING_ASSERT_OP(!=)

#undef KAMPING_ASSERT_OP

private:
    /// @brief Boolean result of this expression.
    bool _result;
    /// @brief Decomposed left hand side of this expression.
    LhsT const& _lhs;
    /// @brief Stringified operand or relation symbol.
    std::string_view _op;
    /// @brief Right hand side of this expression.
    RhsT const& _rhs;
};

/// @brief Decomposed unary expression.
/// @tparam Lhst Decomposed expression type.
template <typename LhsT>
class UnaryExpression : public Expression {
public:
    /// @brief Constructs this unary expression from an expression.
    /// @param lhs The expression.
    explicit UnaryExpression(LhsT const& lhs) : _lhs(lhs) {}

    /// @brief Evaluates this expression.
    /// @return The boolean result of this expression.
    [[nodiscard]] bool result() const final {
        return static_cast<bool>(_lhs);
    }

    /// @brief Writes this expression with stringified operands to the given assertion logger.
    /// @param out The assertion logger.
    void stringify(OStreamLogger& out) const final {
        stringify_value(out, _lhs);
    }

private:
    /// @brief The expression.
    LhsT const& _lhs;
};

/// @brief The left hand size of a decomposed expression. This can either be turned into a \c BinaryExpr if an operand
/// or relation follows, or into a \c UnaryExpr otherwise.
/// @tparam LhsT The expression type.
template <typename LhsT>
class LhsExpression {
public:
    /// @brief Constructs this left hand size of a decomposed expression.
    /// @param lhs The wrapped expression.
    explicit LhsExpression(LhsT const& lhs) : _lhs(lhs) {}

    /// @brief Turns this expression into an \c UnaryExpr. This might only be called if the wrapped expression is
    /// implicitly convertible to \c bool.
    /// @return This expression as \c UnaryExpr.
    UnaryExpression<LhsT> make_unary() {
        static_assert(std::is_convertible_v<LhsT, bool>, "expression must be convertible to bool");
        return UnaryExpression<LhsT>{_lhs};
    }

    // Overload binary operators to return a proxy object that decomposes the rhs of the operator.
#define KAMPING_ASSERT_OP(op)                                                               \
    template <typename RhsT>                                                                \
    friend BinaryExpression<LhsT, RhsT> operator op(LhsExpression&& lhs, RhsT const& rhs) { \
        using namespace std::string_view_literals;                                          \
        return {lhs._lhs op rhs, lhs._lhs, #op##sv, rhs};                                   \
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
    /// @brief The wrapped expression.
    LhsT const& _lhs;
};

/// @brief Decomposes an expression (see group description).
struct Decomposer {
    /// @brief Decomposes an expression (see group description).
    /// @tparam LhsT The type of the expression.
    /// @param lhs The left hand side of the expression.
    /// @return \c lhs wrapped in a \c LhsExpr.
    template <typename LhsT>
    friend LhsExpression<LhsT> operator<=(Decomposer&&, LhsT const& lhs) {
        return LhsExpression<LhsT>(lhs);
    }
};

/// @brief Transforms \c LhsExpression into \c UnaryExpression, does nothing to a \c Expression (see group description).
/// @tparam ExprT Type of the expression, either \c LhsExpression or a \c BinaryExpression.
/// @param expr The expression.
/// @return The expression as some subclass of \c Expression.
template <typename ExprT>
decltype(auto) finalize_expr(ExprT&& expr) {
    if constexpr (std::is_base_of_v<Expression, std::remove_reference_t<std::remove_const_t<ExprT>>>) {
        return std::forward<ExprT>(expr);
    } else {
        return expr.make_unary();
    }
}

/// @}

/// @brief Describes a source code location.
struct SourceLocation {
    /// @brief Filename.
    char const* file;
    /// @brief Line number.
    unsigned row;
    /// @brief Function name.
    char const* function;
};

/// @brief Checks if a assertion of the given level is enabled. This is controlled by the CMake option
/// \c KAMPING_ASSERTION_LEVEL.
/// @param level The level of the assertion.
/// @return Whether the assertion is enabled.
constexpr bool assertion_enabled(int level) {
    return level <= KAMPING_ASSERTION_LEVEL;
}

/// @brief Evaluates an assertion expression. If the assertion fails, prints an error describing the failed assertion.
/// @param type Actual type of this check. In exception mode, this parameter has always value \c ASSERTION, otherwise
/// it names the type of the exception that would have been thrown.
/// @param expr Assertion expression to be checked.
/// @param where Source code location of the assertion.
/// @param expr_str Stringified assertion expression.
/// @return Result of the assertion. If true, the assertion was triggered and the program should be halted.
bool evaluate_and_print_assertion(
    char const* type, Expression&& expr, SourceLocation const& where, char const* expr_str) {
    if (!expr.result()) {
        OStreamLogger(std::cerr) << where.file << ": In function '" << where.function << "':\n"
                                 << where.file << ":" << where.row << ": FAILED " << type << "\n"
                                 << "\t" << expr_str << "\n"
                                 << "with expansion:\n"
                                 << "\t" << expr << "\n";
    }
    return expr.result();
}

/// @endinternal
} // namespace internal
} // namespace kamping::assert
