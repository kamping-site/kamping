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

// overwrite build options and set assertion level to normal
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL kamping::assert::normal

#include "kamping/kassert.hpp"

#include <gmock/gmock.h>

using namespace ::testing;

// General comment: all KASSERT() and KTHROW() calls with a relation in their expression are placed inside lambdas,
// which are then called from EXPECT_EXIT(). This indirection is necessary as otherwise, GCC does not suppress the
// warning on missing parentheses. This happens whenever the KASSERT() call is passed through two levels of macros,
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

TEST(KassertTest, kthrow_overloads_compile) {
    // test that all KTHROW() overloads compile
    EXPECT_THROW(
        { KTHROW(false, "__false_is_false_3__", kamping::assert::KassertException); },
        kamping::assert::KassertException);
    EXPECT_THROW({ KTHROW(false, "__false_is_false_2__"); }, kamping::assert::KassertException);
    EXPECT_THROW({ KTHROW(false); }, kamping::assert::KassertException);
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
    EXPECT_EXIT({ generic_logical_or(false, false); }, KilledBySignal(SIGABRT), "0 || 0");

    EXPECT_EXIT({ generic_logical_and(0, 10); }, KilledBySignal(SIGABRT), "0 && 10"); // implicitly convertible to bool
    EXPECT_EXIT({ generic_logical_or(0, 0); }, KilledBySignal(SIGABRT), "0 || 0");    // implicitly convertible to bool

    // more complex expressions
    auto generic_logical_and_and_and = [](auto const val1, auto const val2, auto const val3, auto const val4) {
        KASSERT(val1 && val2 && val3 && val4);
    };
    auto generic_logical_eq_or_or = [](auto const val1, auto const val2, auto const val3, auto const val4) {
        KASSERT(val1 == val2 || val3 || val4);
    };

    EXPECT_EXIT({ generic_logical_and_and_and(true, false, 10, -1); }, KilledBySignal(SIGABRT), "1 && 0 && 10 && -1");
    EXPECT_EXIT({ generic_logical_eq_or_or(1, 2, false, 0); }, KilledBySignal(SIGABRT), "1 == 2 || 0 || 0");

    // relation + logical operator (more complex expressions on the rhs of the logical operator cannot be decomposed)
    auto generic_eq_and = [](auto const eq_lhs, auto const eq_rhs, auto const and_rhs) {
        KASSERT(eq_lhs == eq_rhs && and_rhs);
    };
    auto generic_lt_or = [](auto const lt_lhs, auto const lt_rhs, auto const or_rhs) {
        KASSERT(lt_lhs < lt_rhs || or_rhs);
    };

    EXPECT_EXIT({ generic_eq_and(1, 2, true); }, KilledBySignal(SIGABRT), "1 == 2 && 1");
    EXPECT_EXIT({ generic_lt_or(2, 1, false); }, KilledBySignal(SIGABRT), "2 < 1 || 0");
}

// Test expression expansion of library-supported types

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

TEST(KassertTest, int_vector_int_pair_expensaion) {
    std::pair<std::vector<int>, int> lhs = {{}, 0};
    std::pair<std::vector<int>, int> rhs = {{1}, 1};

    auto eq = [&] {
        KASSERT(lhs == rhs);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "\\(\\[\\], 0\\) == \\(\\[1\\], 1\\)");
}

// Test expansion of unsupported custom type

TEST(KassertTest, unsupported_type_expansion) {
    struct A {
        bool operator==(A const&) const {
            return false;
        }

        bool operator==(int) const {
            return false;
        }
    };

    auto eq = [] {
        KASSERT(A{} == A{});
    };
    auto eq_int = [](int const val) {
        KASSERT(A{} == val);
    };

    EXPECT_EXIT({ eq(); }, KilledBySignal(SIGABRT), "<\\?> == <\\?>");
    EXPECT_EXIT({ eq_int(42); }, KilledBySignal(SIGABRT), "<\\?> == 42");
}
