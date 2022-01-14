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

// Test that macros produce valid code

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
    const bool var_true  = true;
    const bool var_false = false;
    KASSERT(var_true);
    KASSERT(!var_false);

    // function calls
    auto id = [](const bool ans) {
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
    const bool var_true  = true;
    const bool var_false = false;
    EXPECT_EXIT({ KASSERT(var_false); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(!var_true); }, KilledBySignal(SIGABRT), "");

    // functions
    auto id = [](const bool ans) {
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
    EXPECT_EXIT({ KASSERT(1 != 1); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(1 == 2); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(1 < 1); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(1 > 1); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(2 <= 1); }, KilledBySignal(SIGABRT), "");
    EXPECT_EXIT({ KASSERT(1 >= 2); }, KilledBySignal(SIGABRT), "");
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

// Test expression expansion of library-supported types

TEST(KassertTest, empty_and_single_int_vector_expansion) {
    std::vector<int> lhs = {};
    std::vector<int> rhs = {0};
    EXPECT_EXIT({ KASSERT(lhs == rhs); }, KilledBySignal(SIGABRT), "\\[\\] == \\[0\\]");
}

TEST(KassertTest, multi_element_int_vector_expansion) {
    std::vector<int> lhs = {1, 2, 3};
    std::vector<int> rhs = {1, 2};
    EXPECT_EXIT({ KASSERT(lhs == rhs); }, KilledBySignal(SIGABRT), "\\[1, 2, 3\\] == \\[1, 2\\]");
}

TEST(KassertTest, int_int_pair_expansion) {
    std::pair<int, int> lhs = {1, 2};
    std::pair<int, int> rhs = {1, 3};
    EXPECT_EXIT({ KASSERT(lhs == rhs); }, KilledBySignal(SIGABRT), "\\(1, 2\\) == \\(1, 3\\)");
}

TEST(KassertTest, int_int_pair_vector_expansion) {
    std::vector<std::pair<int, int>> lhs = {{1, 2}, {1, 3}};
    std::vector<std::pair<int, int>> rhs = {{1, 2}, {1, 4}};
    EXPECT_EXIT(
        { KASSERT(lhs == rhs); }, KilledBySignal(SIGABRT),
        "\\[\\(1, 2\\), \\(1, 3\\)\\] == \\[\\(1, 2\\), \\(1, 4\\)\\]");
}

TEST(KassertTest, int_vector_int_pair_expensaion) {
    std::pair<std::vector<int>, int> lhs = {{}, 0};
    std::pair<std::vector<int>, int> rhs = {{1}, 1};
    EXPECT_EXIT({ KASSERT(lhs == rhs); }, KilledBySignal(SIGABRT), "\\(\\[\\], 0\\) == \\(\\[1\\], 1\\)");
}

// Test expansion of unsupported custom type

TEST(KassertTest, unsupported_type_expansion) {
    struct A {
        bool operator==(A const&) const {
            return false;
        }
    };

    EXPECT_EXIT({ KASSERT(A{} == A{}); }, KilledBySignal(SIGABRT), "<\\?> == <\\?>");
}