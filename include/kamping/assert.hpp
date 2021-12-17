/// @file
/// @brief Macros for optional runtime checks

#pragma once

#include <ostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#define KAMPING_ASSERTION_LEVEL 1
#define KAMPING_EXCEPTION_MODE  1

#define KAMPING_SOURCE_LOCATION                 \
    kamping::assert::internal::SourceLocation { \
        __FILE__, __LINE__, __func__            \
    }

#define KAMPING_ASSERT(expression, message, level)                                                                   \
    do {                                                                                                             \
        if constexpr (kamping::assert::internal::assertion_enabled(level)) {                                         \
            if (!kamping::assert::internal::evaluate_assertion(                                                      \
                    kamping::assert::internal::finalize_expr(kamping::assert::internal::Decomposer{} <= expression), \
                    KAMPING_SOURCE_LOCATION, #expression)) {                                                         \
                kamping::assert::Logger(std::cerr) << message << "\n";                                               \
                std::abort();                                                                                        \
            }                                                                                                        \
        }                                                                                                            \
    } while (false)

namespace kamping::assert {
namespace internal {
template <typename, typename, typename = void>
struct IsPrintableType : std::false_type {};

template <typename StreamT, typename ValueT>
struct IsPrintableType<StreamT, ValueT, std::void_t<decltype(std::declval<StreamT&>() << std::declval<ValueT>())>>
    : std::true_type {};
} // namespace internal

class Logger {
public:
    explicit Logger(std::ostream& out) : _out(out) {}

    template <typename T, std::enable_if_t<internal::IsPrintableType<std::ostream, T>::value, int> = 0>
    Logger& operator<<(T&& value) {
        _out << std::forward<T>(value);
        return *this;
    }

private:
    std::ostream& _out;
};

template <typename ValueT, typename AllocatorT>
Logger& operator<<(Logger& logger, std::vector<ValueT, AllocatorT> const& container) {
    if (container.empty()) {
        return logger << "<>";
    }

    logger << "<";
    for (const auto& element: container) {
        logger << element << ", ";
    }
    return logger << "\b\b>";
}

template <typename Key, typename Value>
Logger& operator<<(Logger& logger, std::pair<Key, Value> const& pair) {
    return logger << "<" << pair.first << ", " << pair.second << ">";
}

namespace internal {
template <typename T>
void stringify_value(Logger& out, const T& value) {
    if constexpr (IsPrintableType<Logger, T>::value) {
        out << value;
    } else {
        out << "<?>";
    }
}

class Expr {
public:
    virtual ~Expr() = default;

    [[nodiscard]] virtual bool result() const = 0;

    virtual void stringify(Logger& out) const = 0;

    friend Logger& operator<<(Logger& out, Expr const& expr) {
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

    void stringify(Logger& out) const final {
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

    void stringify(Logger& out) const final {
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

bool evaluate_assertion(Expr&& expr, const SourceLocation& where, char const* expr_str) {
    if (!expr.result()) {
        Logger(std::cerr) << where.file << ": In function '" << where.function << "':\n"
                          << where.file << ":" << where.row << ": FAILED ASSERTION\n"
                          << "\t" << expr_str << "\n"
                          << "with expansion:\n"
                          << "\t" << expr << "\n";
    }
    return expr.result();
}
} // namespace internal

// predefined assertion levels
constexpr int lightweight = 1;
constexpr int normal      = 2;
constexpr int heavy       = 3;
} // namespace kamping::assert
