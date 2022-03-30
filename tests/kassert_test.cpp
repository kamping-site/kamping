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

// Overwrite build option and set assertion level to normal
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL 3

#include "helpers_for_testing.hpp"
#include "kamping/kassert.hpp"

#include <gmock/gmock.h>

using namespace ::testing;

// General comment: all KASSERT() and THROWING_KASSERT() calls with a relation in their expression are placed inside
// lambdas, which are then called from EXPECT_EXIT(). This indirection is necessary as otherwise, GCC does not suppress
// the warning on missing parentheses. This happens whenever the KASSERT() call is passed through two levels of macros,
// i.e.,
//
// #defined A(stmt) B(stmt)
// #defined B(stmt) stmt;
//
// A(KASSERT(false)); // warning not suppressed (with GCC only)
// B(KASSERT(false)); // warning suppress, code ok

TEST(KassertTest, kassert_overloads_compile) {
    // test that all KASSERT overloads compile
    EXPECT_EXIT(
        { KASSERT(false, "__false_is_false_3__", kamping::assert::normal); }, KilledBySignal(SIGABRT),
        "__false_is_false_3__");
    EXPECT_EXIT({ KASSERT(false, "__false_is_false_2__"); }, KilledBySignal(SIGABRT), "__false_is_false_2__");
    EXPECT_EXIT({ KASSERT(false); }, KilledBySignal(SIGABRT), "");
}

TEST(KassertTestingHelperTest, kassert_testing_helper) {
    auto failing_function = [] {
        KASSERT(false, "__false_is_false_1__");
    };

    // Pass a single function call to the macro.
    EXPECT_KASSERT_FAILS(failing_function(), "__false_is_false_1");
    ASSERT_KASSERT_FAILS(failing_function(), "__false_is_false_1");

    // Pass a code block to the macro.
    EXPECT_KASSERT_FAILS({ failing_function(); }, "__false_is_false_1");
    ASSERT_KASSERT_FAILS({ failing_function(); }, "__false_is_false_1");
}

// Since we explicitly set the assertion level to normal, heavier assertions should not trigger.
TEST(KassertTest, kassert_respects_assertion_level) {
    EXPECT_EXIT({ KASSERT(false, "", kamping::assert::light); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(false, "", kamping::assert::normal); }, KilledBySignal(SIGABRT), "");
    KASSERT(false, "", kamping::assert::light_communication);
    KASSERT(false, "", kamping::assert::heavy_communication);
    KASSERT(false, "", kamping::assert::heavy);
}

TEST(KassertTest, kthrow_overloads_compile) {
#ifdef KAMPING_EXCEPTION_MODE
    // test that all THROWING_KASSERT() overloads compile
    EXPECT_THROW({ THROWING_KASSERT(false, "__false_is_false_2__"); }, kamping::KassertException);
    EXPECT_THROW({ THROWING_KASSERT(false); }, kamping::KassertException);
#else  // KAMPING_EXCEPTION_MODE
    EXPECT_EXIT({ THROWING_KASSERT(false, "__false_is_false_2__"); }, KilledBySignal(SIGABRT), "__false_is_false_2__");
    EXPECT_EXIT({ THROWING_KASSERT(false); }, KilledBySignal(SIGABRT), "");
#endif // KAMPING_EXCEPTION_MODE
}

class ZeroCustomArgException : public std::exception {
public:
    ZeroCustomArgException(std::string) {}

    const char* what() const throw() final {
        return "ZeroCustomArgException";
    }
};

class SingleCustomArgException : public std::exception {
public:
    SingleCustomArgException(std::string, int) {}

    const char* what() const throw() final {
        return "SingleCustomArgException";
    }
};

TEST(KassertTest, kthrow_custom_compiles) {
#ifdef KAMPING_EXCEPTION_MODE
    EXPECT_THROW({ THROWING_KASSERT_SPECIFIED(false, "", ZeroCustomArgException); }, ZeroCustomArgException);
    EXPECT_THROW({ THROWING_KASSERT_SPECIFIED(false, "", SingleCustomArgException, 43); }, SingleCustomArgException);
#else  // KAMPING_EXCEPTION_MODE
    EXPECT_EXIT(
        { THROWING_KASSERT_SPECIFIED(false, "", ZeroCustomArgException); }, KilledBySignal(SIGABRT),
        "ZeroCustomArgException");
    EXPECT_EXIT(
        { THROWING_KASSERT_SPECIFIED(false, "", SingleCustomArgException, 43); }, KilledBySignal(SIGABRT),
        "SingleCustomArgException");
#endif // KAMPING_EXCEPTION_MODE
}

// Check that THROWING_KASSERT does nothing if the expression evaluates to true.
TEST(KassertTest, kthrow_does_nothing_on_true_expression) {
    THROWING_KASSERT(true);
    THROWING_KASSERT(true, "");
    THROWING_KASSERT_SPECIFIED(true, "", ZeroCustomArgException);
}

// Test that expressions are evaluated as expected
// The following tests do not check the expression expansion!

TEST(KassertTest, unary_true_expressions) {
    // unary expressions that evaluate to true and thus should not trigger the assertions

    // literals
    KASSERT(true);
    KASSERT(!false);

    // variables
    bool const var_true  = true;
    bool const var_false = false;
    KASSERT(var_true);
    KASSERT(!var_false);

    // function calls
    auto id = [](bool const ans) {
        return ans;
    };
    KASSERT(id(true));
    KASSERT(!id(false));

    // unary expressions with implicit conversion to true
    KASSERT(10);
    KASSERT(-10);
    KASSERT(1 + 1); // unary expression from KASSERT perspective
}

TEST(KassertTest, unary_false_expressions) {
    // test unary expressions that evaluate to false and should thus trigger the assertion

    // literals
    EXPECT_EXIT({ KASSERT(false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(!true); }, KilledBySignal(SIGABRT), "");

    // variables
    bool const var_true  = true;
    bool const var_false = false;
    EXPECT_EXIT({ KASSERT(var_false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(!var_true); }, KilledBySignal(SIGABRT), "");

    // functions
    auto id = [](bool const ans) {
        return ans;
    };
    EXPECT_EXIT({ KASSERT(id(false)); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(!id(true)); }, KilledBySignal(SIGABRT), "");

    // expressions implicitly convertible to bool
    EXPECT_EXIT({ KASSERT(0); }, KilledBySignal(SIGABRT), "");
    // EXPECT_EXIT({ KASSERT(nullptr); }, KilledBySignal(SIGABRT), ""); -- std::nullptr_t is not convertible to bool
    EXPECT_EXIT({ KASSERT(1 - 1); }, KilledBySignal(SIGABRT), ""); // unary expression from KASSERT perspective
}

TEST(KassertTest, true_arithmetic_relation_expressions) {
    KASSERT(1 == 1);
    KASSERT(1 != 2);
    KASSERT(1 < 2);
    KASSERT(2 > 1);
    KASSERT(1 <= 2);
    KASSERT(2 >= 1);
}

TEST(KassertTest, true_logical_operator_expressions) {
    KASSERT(true && true);
    KASSERT(true && true && true);
    KASSERT((true && true) && true);
    KASSERT(true && (true && true));
    KASSERT(true || false);
    KASSERT(false || true);
    KASSERT((true && false) || true);
    KASSERT(true || (false && true));
    KASSERT(!false || false);
    KASSERT(true && !false);
}

TEST(KassertTest, false_arithmetic_relation_expressions) {
    auto eq = [] {
        KASSERT(1 == 2);
    };
    auto neq = [] {
        KASSERT(1 != 1);
    };
    auto lt = [] {
        KASSERT(1 < 1);
    };
    auto gt = [] {
        KASSERT(1 > 1);
    };
    auto le = [] {
        KASSERT(2 <= 1);
    };
    auto ge = [] {
        KASSERT(1 >= 2);
    };
    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ neq(); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ lt(); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ gt(); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ le(); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ ge(); }, KilledBySignal(SIGABRT), "");
}

TEST(KassertTest, false_logical_operator_expressions) {
    EXPECT_EXIT({ KASSERT(true && false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(true && (true && false)); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(true && (false || false)); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(false || (true && false)); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(false && true); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(false || false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(!false && false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(false && !false); }, KilledBySignal(SIGABRT), "");
}

TEST(KassertTest, true_chained_relation_ops) {
    KASSERT(1 == 1 == 1);
    KASSERT(1 == 1 != 0);
    KASSERT(1 == 1 & 1);
    KASSERT(5 == 0 | 1);
    KASSERT(5 == 0 ^ 1);
    KASSERT(5 == 5 ^ false);
}

// Test expression expansion of primitive types

TEST(KassertTest, primitive_type_expansion) {
    // arithmetic operators
    auto generic_eq = [](auto const lhs, auto const rhs) {
        KASSERT(lhs == rhs);
    };
    auto generic_gt = [](auto const lhs, auto const rhs) {
        KASSERT(lhs > rhs);
    };
    auto generic_ge = [](auto const lhs, auto const rhs) {
        KASSERT(lhs >= rhs);
    };
    auto generic_lt = [](auto const lhs, auto const rhs) {
        KASSERT(lhs < rhs);
    };
    auto generic_le = [](auto const lhs, auto const rhs) {
        KASSERT(lhs <= rhs);
    };

    EXPECT_EXIT({ generic_eq(1, 2); }, KilledBySignal(SIGABRT), "1 == 2");
    EXPECT_EXIT({ generic_gt(1, 2); }, KilledBySignal(SIGABRT), "1 > 2");
    EXPECT_EXIT({ generic_ge(1, 2); }, KilledBySignal(SIGABRT), "1 >= 2");
    EXPECT_EXIT({ generic_lt(2, 1); }, KilledBySignal(SIGABRT), "2 < 1");
    EXPECT_EXIT({ generic_le(2, 1); }, KilledBySignal(SIGABRT), "2 <= 1");

    // logical operators
    auto generic_logical_and = [](auto const lhs, auto const rhs) {
        KASSERT(lhs && rhs);
    };
    auto generic_logical_or = [](auto const lhs, auto const rhs) {
        KASSERT(lhs || rhs);
    };

    EXPECT_EXIT({ generic_logical_and(true, false); }, KilledBySignal(SIGABRT), "1 && 0");
    EXPECT_EXIT({ generic_logical_or(false, false); }, KilledBySignal(SIGABRT), "0 \\|\\| 0");

    EXPECT_EXIT({ generic_logical_and(0, 10); }, KilledBySignal(SIGABRT), "0 && 10");  // implicitly convertible to bool
    EXPECT_EXIT({ generic_logical_or(0, 0); }, KilledBySignal(SIGABRT), "0 \\|\\| 0"); // implicitly convertible to bool

    // more complex expressions
    auto generic_logical_and_and_and = [](auto const val1, auto const val2, auto const val3, auto const val4) {
        KASSERT(val1 && val2 && val3 && val4);
    };
    auto generic_logical_eq_or_or = [](auto const val1, auto const val2, auto const val3, auto const val4) {
        KASSERT(val1 == val2 || val3 || val4);
    };

    EXPECT_EXIT({ generic_logical_and_and_and(true, false, 10, -1); }, KilledBySignal(SIGABRT), "1 && 0 && 10 && -1");
    EXPECT_EXIT({ generic_logical_eq_or_or(1, 2, false, 0); }, KilledBySignal(SIGABRT), "1 == 2 \\|\\| 0 \\|\\| 0");

    // relation + logical operator (more complex expressions on the rhs of the logical operator cannot be decomposed)
    auto generic_eq_and = [](auto const eq_lhs, auto const eq_rhs, auto const and_rhs) {
        KASSERT(eq_lhs == eq_rhs && and_rhs);
    };
    auto generic_lt_or = [](auto const lt_lhs, auto const lt_rhs, auto const or_rhs) {
        KASSERT(lt_lhs < lt_rhs || or_rhs);
    };

    EXPECT_EXIT({ generic_eq_and(1, 2, true); }, KilledBySignal(SIGABRT), "1 == 2 && 1");
    EXPECT_EXIT({ generic_lt_or(2, 1, false); }, KilledBySignal(SIGABRT), "2 < 1 \\|\\| 0");
}

TEST(KassertTest, primitive_type_expansion_limitations) {
    // test expression expansion where the expression cannot be fully expanded

    KASSERT(true && false || true);
    KASSERT(true && true || false);
    KASSERT(false || true && true);
    KASSERT(true || true && false);
    KASSERT(!false || false && false);
    KASSERT(!true || !false && true);

    auto generic_and_or = [](auto const and_rhs, auto const or_rhs, auto const or_lhs) {
        KASSERT(and_rhs && or_rhs || or_lhs);
    };
    auto generic_or_and = [](auto const or_rhs, auto const and_rhs, auto const and_lhs) {
        KASSERT(or_rhs || and_rhs && and_lhs);
    };
    auto generic_neg_or_and = [](auto const neg, auto const and_rhs, auto const and_lhs) {
        KASSERT(!neg || and_rhs && and_lhs);
    };
    auto generic_and_neg_or = [](auto const and_rhs, auto const neg, auto const or_lhs) {
        KASSERT(and_rhs && !neg || or_lhs);
    };

    EXPECT_EXIT({ generic_and_or(true, false, false); }, KilledBySignal(SIGABRT), "1 && 0"); // cannot expand rhs of &&
    EXPECT_EXIT(
        { generic_or_and(false, true, false); }, KilledBySignal(SIGABRT), "0 \\|\\| 0");  // cannot expand rhs of ||
    EXPECT_EXIT({ generic_neg_or_and(5, 1, 0); }, KilledBySignal(SIGABRT), "0 \\|\\| 0"); // cannot expand !, rhs of ||
    EXPECT_EXIT({ generic_and_neg_or(1, 1, false); }, KilledBySignal(SIGABRT), "1 && 0"); // ditto

    // negation + relation
    auto generic_neg_eq = [](const auto lhs_neg, const auto rhs) {
        KASSERT(!lhs_neg == rhs);
    };

    EXPECT_EXIT({ generic_neg_eq(5, 10); }, KilledBySignal(SIGABRT), "0 == 10"); // cannot expand !lhs_neg
}

TEST(KassertTest, chained_rel_ops_expansion) {
    auto generic_chained_eq = [](auto const val1, auto const val2, auto const val3) {
        KASSERT(val1 == val2 == val3);
    };
    auto generic_chained_eq_neq = [](auto const val1, auto const val2, auto const val3) {
        KASSERT(val1 == val2 != val3);
    };
    auto generic_chained_eq_binary_and = [](auto const val1, auto const val2, auto const val3) {
        KASSERT(val1 == val2 & val3);
    };
    auto generic_chained_eq_binary_or = [](auto const val1, auto const val2, auto const val3) {
        KASSERT(val1 == val2 | val3);
    };
    auto generic_chained_eq_binary_xor = [](auto const val1, auto const val2, auto const val3) {
        KASSERT(val1 == val2 ^ val3);
    };

    EXPECT_EXIT({ generic_chained_eq(1, 1, 5); }, KilledBySignal(SIGABRT), "1 == 1 == 5");
    EXPECT_EXIT({ generic_chained_eq_neq(1, 1, 1); }, KilledBySignal(SIGABRT), "1 == 1 != 1");
    EXPECT_EXIT({ generic_chained_eq_binary_and(5, 5, 0); }, KilledBySignal(SIGABRT), "5 == 5 & 0");
    EXPECT_EXIT({ generic_chained_eq_binary_or(5, 4, 0); }, KilledBySignal(SIGABRT), "5 == 4 \\| 0");
    EXPECT_EXIT({ generic_chained_eq_binary_xor(5, 4, 0); }, KilledBySignal(SIGABRT), "5 == 4 \\^ 0");
}

// Test expression expansion of library-supported types

TEST(KassertTest, true_complex_expanded_types) {
    std::vector<int> vec_rhs = {1, 2, 3};
    std::vector<int> vec_lhs = {1, 2, 3};
    KASSERT(vec_rhs == vec_lhs);

    std::pair<int, std::vector<int>> pair_vec_rhs = {1, {2, 3}};
    std::pair<int, std::vector<int>> pair_vec_lhs = {1, {2, 3}};
    KASSERT(pair_vec_rhs == pair_vec_lhs);
}

TEST(KassertTest, empty_and_single_int_vector_expansion) {
    std::vector<int> lhs = {};
    std::vector<int> rhs = {0};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\[\\] == \\[0\\]");
}

TEST(KassertTest, multi_element_int_vector_expansion) {
    std::vector<int> lhs = {1, 2, 3};
    std::vector<int> rhs = {1, 2};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\[1, 2, 3\\] == \\[1, 2\\]");
}

TEST(KassertTest, int_int_pair_expansion) {
    std::pair<int, int> lhs = {1, 2};
    std::pair<int, int> rhs = {1, 3};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };
    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\(1, 2\\) == \\(1, 3\\)");
}

TEST(KassertTest, int_int_pair_vector_expansion) {
    std::vector<std::pair<int, int>> lhs = {{1, 2}, {1, 3}};
    std::vector<std::pair<int, int>> rhs = {{1, 2}, {1, 4}};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\[\\(1, 2\\), \\(1, 3\\)\\] == \\[\\(1, 2\\), \\(1, 4\\)\\]");
}

TEST(KassertTest, int_vector_int_pair_expansion) {
    std::pair<std::vector<int>, int> lhs = {{}, 0};
    std::pair<std::vector<int>, int> rhs = {{1}, 1};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\(\\[\\], 0\\) == \\(\\[1\\], 1\\)");
}

// Test expansion of unsupported custom type

TEST(KassertTest, unsupported_type_expansion) {
    struct CustomType {
        bool operator==(CustomType const&) const {
            return false;
        }

        bool operator==(int) const {
            return false;
        }
    };

    auto eq = [] {
        KASSERT(CustomType{} == CustomType{});
    };
    auto eq_int = [](int const val) {
        KASSERT(CustomType{} == val);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "<\\?> == <\\?>");
    EXPECT_EXIT({ eq_int(42); }, KilledBySignal(SIGABRT), "<\\?> == 42");
}
