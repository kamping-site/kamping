/// @file
/// @brief Macros for optional runtime checks

#pragma once

#include <ostream>
#include <string_view>
#include <type_traits>

#define KAMPING_ASSERTION_LEVEL 1
#define KAMPING_EXCEPTION_MODE  1

namespace kamping {
namespace assert {
namespace internal {
template <typename, typename = void>
struct IsPrintableType : std::false_type {};

template <typename ValueT>
struct IsPrintableType<ValueT, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<ValueT>())>>
    : std::true_type {};

template <typename T>
void stringify_value(std::ostream& out, const T& value) {
    if constexpr (IsPrintableType<T>::value) {
        out << value;
    } else {
        out << "<?>";
    }
}

class Expr {
public:
    virtual ~Expr() = default;

    [[nodiscard]] virtual bool result() const = 0;

    virtual void stringify(std::ostream& out) const = 0;

    friend std::ostream& operator<<(std::ostream& out, Expr const& expr) {
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

    void stringify(std::ostream& out) const final {
        stringify_value(out, _lhs);
        out << " " << _op << " ";
        stringify_value(out, _rhs);
    }

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

    void stringify(std::ostream& out) const final {
        stringify_value(out, _lhs);
    }

private:
    const LhsT& _lhs;
};

template <typename LhsT>
class LhsExpr {
public:
    explicit LhsExpr(LhsT const& lhs) : _lhs(lhs) {}

    template <typename RhsT>
    friend BinaryExpr<LhsT, RhsT> operator==(LhsExpr&& lhs, RhsT const& rhs) {
        using namespace std::string_view_literals;
        return {lhs._lhs == rhs, lhs._lhs, "=="sv, rhs};
    }

    template <typename RhsT>
    friend BinaryExpr<LhsT, RhsT> operator!=(LhsExpr&& lhs, RhsT const& rhs) {
        using namespace std::string_view_literals;
        return {lhs._lhs != rhs, lhs._lhs, "!="sv, rhs};
    }

    UnaryExpr<LhsT> make_unary() {
        static_assert(std::is_convertible_v<LhsT, bool>, "expression must be convertible to bool");
        return {_lhs};
    }

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
        std::cerr << where.file << ": In function '" << where.function << "':\n"
                  << where.file << ":" << where.row << ": Lordy, lordy, look who's faulty:\n"
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
} // namespace assert

#define SOURCE_LOCATION                         \
    kamping::assert::internal::SourceLocation { \
        __FILE__, __LINE__, __func__            \
    }

#define ASSERT(expression, message, level)                                                                           \
    do {                                                                                                             \
        if constexpr (kamping::assert::internal::assertion_enabled(level)) {                                         \
            if (!kamping::assert::internal::evaluate_assertion(                                                      \
                    kamping::assert::internal::finalize_expr(kamping::assert::internal::Decomposer{} <= expression), \
                    SOURCE_LOCATION, #expression)) {                                                                 \
                std::abort();                                                                                        \
            }                                                                                                        \
        }                                                                                                            \
    } while (false)
} // namespace kamping
