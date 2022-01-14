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

TEST(KassertTest, unary_true_expressions) {
    // unary expressions that evaluate to true and thus should not trigger the assertions

    // literals
    KASSERT(true);
    KASSERT(!false);

    // variables
    const bool var_true = true;
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
    EXPECT_EXIT({ KASSERT(false); }, KilledBySignal(SIGABRT), "0");
    EXPECT_EXIT({ KASSERT(!true); }, KilledBySignal(SIGABRT), "0");

    // variables
    const bool var_true = true;
    const bool var_false = false;
    EXPECT_EXIT({ KASSERT(var_false); }, KilledBySignal(SIGABRT), "0");
    EXPECT_EXIT({ KASSERT(!var_true); }, KilledBySignal(SIGABRT), "0");

    // functions
    auto id = [](const bool ans) {
        return ans;
    };
    EXPECT_EXIT({ KASSERT(id(false)); }, KilledBySignal(SIGABRT), "0");
    EXPECT_EXIT({ KASSERT(!id(true)); }, KilledBySignal(SIGABRT), "0");

    // expressions implicitly convertible to bool
    EXPECT_EXIT({ KASSERT(0); }, KilledBySignal(SIGABRT), "0");
    // EXPECT_EXIT({ KASSERT(nullptr); }, KilledBySignal(SIGABRT), "0"); -- std::nullptr_t is not convertible to bool
    EXPECT_EXIT({ KASSERT(1 - 1); }, KilledBySignal(SIGABRT), "0"); // unary expression from KASSERT perspective
}

TEST(KassertTest, kassert_overloads_compile) {
    EXPECT_EXIT(
        { KASSERT(false, "__false_is_false_3__", kamping::assert::normal); }, KilledBySignal(SIGABRT),
        "__false_is_false_3__");
    EXPECT_EXIT({ KASSERT(false, "__false_is_false_2__"); }, KilledBySignal(SIGABRT), "__false_is_false_2__");
    EXPECT_EXIT({ KASSERT(false); }, KilledBySignal(SIGABRT), "");
}

TEST(KassertTest, kthrow_overloads_compile) {
    EXPECT_THROW(
        { KTHROW(false, "__false_is_false_3__", kamping::assert::KassertException); },
        kamping::assert::KassertException);
    EXPECT_THROW({ KTHROW(false, "__false_is_false_2__"); }, kamping::assert::KassertException);
    EXPECT_THROW({ KTHROW(false); }, kamping::assert::KassertException);
}