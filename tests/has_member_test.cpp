// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>

#include "kamping/has_member.hpp"

namespace kamping::type_traits {
KAMPING_MAKE_HAS_MEMBER(foo);
KAMPING_MAKE_HAS_MEMBER_TEMPLATE(baz);
} // namespace kamping::type_traits

class ClassWithFoo {
public:
    int foo();
};

class ClassWithFooAndArguments {
public:
    int foo(double, char);
};
class ClassWithFooTemplate {
public:
    template <typename T, typename K>
    int foo();
};

class ClassWithBaz {
public:
    void baz();
};

class ClassWithBazTemplate {
public:
    template <typename T = double, typename K = char>
    void baz();
};

class EmptyClass {};

TEST(HasMemberTest, make_has_member_works) {
    EXPECT_TRUE(kamping::type_traits::has_member_foo<ClassWithFoo>::value);
    EXPECT_TRUE(kamping::type_traits::has_member_foo_v<ClassWithFoo>);

    EXPECT_TRUE(kamping::type_traits::has_member_foo<ClassWithFooAndArguments>::value);
    EXPECT_TRUE(kamping::type_traits::has_member_foo_v<ClassWithFooAndArguments>);

    EXPECT_FALSE(kamping::type_traits::has_member_foo<ClassWithFooTemplate>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_foo_v<ClassWithFooTemplate>);

    EXPECT_FALSE(kamping::type_traits::has_member_foo<ClassWithBaz>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_foo_v<ClassWithBaz>);

    EXPECT_FALSE(kamping::type_traits::has_member_foo<ClassWithBazTemplate>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_foo_v<ClassWithBazTemplate>);

    EXPECT_FALSE(kamping::type_traits::has_member_foo<EmptyClass>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_foo_v<EmptyClass>);
}

TEST(HasMemberTest, make_has_member_template) {
    EXPECT_FALSE(kamping::type_traits::has_member_baz<ClassWithBaz>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_baz_v<ClassWithBaz>);

    EXPECT_TRUE(kamping::type_traits::has_member_baz<ClassWithBazTemplate>::value);
    EXPECT_TRUE(kamping::type_traits::has_member_baz_v<ClassWithBazTemplate>);

    EXPECT_TRUE((kamping::type_traits::has_member_baz<ClassWithBazTemplate, int, char>::value));
    EXPECT_TRUE((kamping::type_traits::has_member_baz_v<ClassWithBazTemplate, int, char>));

    EXPECT_FALSE(kamping::type_traits::has_member_baz<EmptyClass>::value);
    EXPECT_FALSE(kamping::type_traits::has_member_baz_v<EmptyClass>);
}
