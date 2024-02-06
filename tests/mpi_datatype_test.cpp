// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/environment.hpp"
#include "kamping/mpi_datatype.hpp"

using namespace ::kamping;
using namespace ::testing;

std::set<MPI_Datatype> freed_types;

int MPI_Type_free(MPI_Datatype* type) {
    freed_types.insert(*type);
    return PMPI_Type_free(type);
}

MATCHER_P2(ContiguousType, type, n, "") {
    int num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(arg, &num_integers, &num_addresses, &num_datatypes, &combiner);
    if (combiner != MPI_COMBINER_CONTIGUOUS) {
        *result_listener << "not a contiguous type";
        return false;
    }
    int          count;
    MPI_Datatype underlying_type;
    MPI_Type_get_contents(arg, num_integers, num_addresses, num_datatypes, &count, nullptr, &underlying_type);
    if (count != n) {
        *result_listener << "wrong count";
        return false;
    }
    PrintToString(type);
    return underlying_type == type;
}

// Returns a std::vector containing all MPI_Datatypes equivalent to the given C++ datatype on this machine.
// Removes the topmost level of const and volatile qualifiers.
template <typename T>
std::vector<MPI_Datatype> possible_mpi_datatypes() noexcept {
    // Remove const qualifiers.
    using T_no_const = std::remove_const_t<T>;

    // Check if we got a array type -> create a continuous type.
    if constexpr (std::is_array_v<T_no_const>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        // return std::vector<MPI_Datatype>{mpi_custom_continuous_type<sizeof(T_no_cv)>()};
        return std::vector<MPI_Datatype>{};
    }

    // Check if we got a enum type -> use underlying type
    if constexpr (std::is_enum_v<T_no_const>) {
        return possible_mpi_datatypes<std::underlying_type_t<T_no_const>>();
    }

    // For each supported C++ datatype, check if it is equivalent to the T_no_cv and if so, add the corresponding MPI
    // datatype to the list of possible types.
    std::vector<MPI_Datatype> possible_mpi_datatypes;
    if constexpr (std::is_same_v<T_no_const, char>) {
        possible_mpi_datatypes.push_back(MPI_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, signed char>) {
        possible_mpi_datatypes.push_back(MPI_SIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned char>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, wchar_t>) {
        possible_mpi_datatypes.push_back(MPI_WCHAR);
    }
    if constexpr (std::is_same_v<T_no_const, signed short>) {
        possible_mpi_datatypes.push_back(MPI_SHORT);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned short>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_SHORT);
    }
    if constexpr (std::is_same_v<T_no_const, signed int>) {
        possible_mpi_datatypes.push_back(MPI_INT);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED);
    }
    if constexpr (std::is_same_v<T_no_const, signed long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, signed long long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned long long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, float>) {
        possible_mpi_datatypes.push_back(MPI_FLOAT);
    }
    if constexpr (std::is_same_v<T_no_const, double>) {
        possible_mpi_datatypes.push_back(MPI_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_const, long double>) {
        possible_mpi_datatypes.push_back(MPI_LONG_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_const, int8_t>) {
        possible_mpi_datatypes.push_back(MPI_INT8_T);
    }
    if constexpr (std::is_same_v<T_no_const, int16_t>) {
        possible_mpi_datatypes.push_back(MPI_INT16_T);
    }
    if constexpr (std::is_same_v<T_no_const, int32_t>) {
        possible_mpi_datatypes.push_back(MPI_INT32_T);
    }
    if constexpr (std::is_same_v<T_no_const, int64_t>) {
        possible_mpi_datatypes.push_back(MPI_INT64_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint8_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT8_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint16_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT16_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint32_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT32_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint64_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT64_T);
    }
    if constexpr (std::is_same_v<T_no_const, bool>) {
        possible_mpi_datatypes.push_back(MPI_CXX_BOOL);
    }
    if constexpr (std::is_same_v<T_no_const, kamping::kabool>) {
        possible_mpi_datatypes.push_back(MPI_CXX_BOOL);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<float>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_FLOAT_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<double>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_DOUBLE_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<long double>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_LONG_DOUBLE_COMPLEX);
    }

    assert(possible_mpi_datatypes.size() > 0);
    return possible_mpi_datatypes;
}

TEST(MpiDataTypeTest, mpi_datatype_basics) {
    // Check using the equivalent_mpi_datatypes() helper.
    EXPECT_THAT(possible_mpi_datatypes<char>(), Contains(mpi_datatype<char>()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned char>(), Contains(mpi_datatype<unsigned char>()));
    EXPECT_THAT(possible_mpi_datatypes<signed char>(), Contains(mpi_datatype<signed char>()));
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_datatype<uint8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<int8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<wchar_t>(), Contains(mpi_datatype<wchar_t>()));
    EXPECT_THAT(possible_mpi_datatypes<short>(), Contains(mpi_datatype<short>()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned short>(), Contains(mpi_datatype<unsigned short>()));
    EXPECT_THAT(possible_mpi_datatypes<signed short>(), Contains(mpi_datatype<signed short>()));
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(mpi_datatype<int>()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned>(), Contains(mpi_datatype<unsigned>()));
    EXPECT_THAT(possible_mpi_datatypes<long>(), Contains(mpi_datatype<long>()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned long>(), Contains(mpi_datatype<unsigned long>()));
    EXPECT_THAT(possible_mpi_datatypes<signed long long int>(), Contains(mpi_datatype<signed long long int>()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned long long int>(), Contains(mpi_datatype<unsigned long long int>()));
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(mpi_datatype<float>()));
    EXPECT_THAT(possible_mpi_datatypes<double>(), Contains(mpi_datatype<double>()));
    EXPECT_THAT(possible_mpi_datatypes<long double>(), Contains(mpi_datatype<long double>()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<int8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int16_t>(), Contains(mpi_datatype<int16_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int32_t>(), Contains(mpi_datatype<int32_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int64_t>(), Contains(mpi_datatype<int64_t>()));
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_datatype<uint8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<uint16_t>(), Contains(mpi_datatype<uint16_t>()));
    EXPECT_THAT(possible_mpi_datatypes<uint32_t>(), Contains(mpi_datatype<uint32_t>()));
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(mpi_datatype<uint64_t>()));
    EXPECT_THAT(possible_mpi_datatypes<bool>(), Contains(mpi_datatype<bool>()));
    EXPECT_THAT(possible_mpi_datatypes<kamping::kabool>(), Contains(mpi_datatype<kamping::kabool>()));
    EXPECT_THAT(possible_mpi_datatypes<std::complex<double>>(), Contains(mpi_datatype<std::complex<double>>()));
    EXPECT_THAT(possible_mpi_datatypes<std::complex<float>>(), Contains(mpi_datatype<std::complex<float>>()));
    EXPECT_THAT(
        possible_mpi_datatypes<std::complex<long double>>(),
        Contains(mpi_datatype<std::complex<long double>>())
    );
}

TEST(MpiDataTypeTest, mpi_datatype_const_and_volatile) {
    // Ignore const and volatile qualifiers.
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<int8_t const>()));
    // EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<int8_t volatile>()));
    // EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<const volatile int8_t>()));
}

TEST(MpiDataTypeTest, mpi_datatype_typedefs_and_using) {
    // typedefs and using directives.
    typedef int myInt;
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(mpi_datatype<myInt>()));

    using myFloat = float;
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(mpi_datatype<myFloat>()));
}

TEST(MpiDataTypeTest, mpi_datatype_size_t) {
    // size_t, which should be one of the unsigned integer types with at least 16 bits (as of C++11).
    EXPECT_THAT(
        (std::array{MPI_UNSIGNED_SHORT, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG}),
        Contains(mpi_datatype<size_t>())
    );

    // As should std::size_t, which should be one of the unsigned integer types with at least 16 bits (as of C++11).
    EXPECT_THAT(
        (std::array{MPI_UNSIGNED_SHORT, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG}),
        Contains(mpi_datatype<std::size_t>())
    );
}

TEST(MpiDataTypeTest, mpi_datatype_enum) {
    // Calling mpi_datatype with a enum type should use the underlying type.

    // Unscoped enum
    enum unscopedEnum { valueA = 0, valueB = 1 };
    auto unscopedEnum_types = possible_mpi_datatypes<std::underlying_type_t<unscopedEnum>>();
    EXPECT_THAT(unscopedEnum_types, Contains(mpi_datatype<unscopedEnum>()));

    // Unscoped enum with explicit underlying type
    enum unscopedEnumInt : int { valueA2 = 0, valueB2 = 1 };
    auto unscopedEnumInt_types = possible_mpi_datatypes<std::underlying_type_t<unscopedEnumInt>>();
    EXPECT_THAT(unscopedEnumInt_types, Contains(mpi_datatype<unscopedEnumInt>()));

    // Scoped enum
    enum class scopedEnum { valueA = 0, valueB = 1 };
    auto scopedEnum_types = possible_mpi_datatypes<std::underlying_type_t<scopedEnum>>();
    EXPECT_THAT(scopedEnum_types, Contains(mpi_datatype<scopedEnum>()));

    // Scope enum with explicit underlying type
    enum class scopedEnumUint8_t : uint8_t { valueA = 0, valueB = 1 };
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_datatype<scopedEnumUint8_t>()));

    enum class scopedEnumInt64_t : int64_t { valueA = 0, valueB = 1 };
    EXPECT_THAT(possible_mpi_datatypes<int64_t>(), Contains(mpi_datatype<scopedEnumInt64_t>()));
}

TEST(MpiDataTypeTest, mpi_datatype_continuous_type) {
    struct TestStruct {
        int a;
        int b;
    };

    // There seems to be no way to check if a given datatype in MPI is a custom type, we therefore rule out that it's
    // equal to any of the other types, including the NULL type.
    EXPECT_NE(MPI_DATATYPE_NULL, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CHAR, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CHAR, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_SIGNED_CHAR, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UNSIGNED_CHAR, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_WCHAR, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_SHORT, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UNSIGNED_SHORT, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_INT, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UNSIGNED, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_LONG, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UNSIGNED_LONG, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_LONG_LONG, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UNSIGNED_LONG_LONG, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_FLOAT, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_DOUBLE, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_LONG_DOUBLE, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_INT8_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_INT16_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_INT32_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_INT64_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UINT8_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UINT16_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UINT32_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_UINT64_T, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CXX_BOOL, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CXX_FLOAT_COMPLEX, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CXX_DOUBLE_COMPLEX, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_CXX_LONG_DOUBLE_COMPLEX, mpi_datatype<TestStruct>());
}

TEST(MpiDataTypeTest, mpi_datatype_c_array) {
    // Calling mpi_datatype with an array should return a continuous datatype.
    {
        int c_array[3];
        EXPECT_THAT(mpi_datatype<decltype(c_array)>(), ContiguousType(MPI_INT, 3));
    }
    {
        int* ptr;
        EXPECT_EQ(mpi_type_traits<decltype(ptr)>::category, TypeCategory::undefined);
    }
    {
        std::array<int, 3> cpp_array;
        EXPECT_THAT(mpi_datatype<decltype(cpp_array)>(), ContiguousType(MPI_INT, 3));
    }

    {
        std::array<double, 3> cpp_array;
        EXPECT_THAT(mpi_datatype<decltype(cpp_array)>(), ContiguousType(MPI_DOUBLE, 3));
    }
}

TEST(MpiDataTypeTest, test_type_groups) {
    struct DummyType {
        int  a;
        char b;
    };
    EXPECT_EQ(kamping::mpi_type_traits<int>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<signed int>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<signed long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<short>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<signed short>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned short>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned int>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<long long int>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<long long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<signed long long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned long long>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<signed char>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<unsigned char>::category, kamping::TypeCategory::integer);

    EXPECT_EQ(kamping::mpi_type_traits<int8_t>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<int16_t>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<int32_t>::category, kamping::TypeCategory::integer);

    EXPECT_EQ(kamping::mpi_type_traits<uint8_t>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<uint16_t>::category, kamping::TypeCategory::integer);
    EXPECT_EQ(kamping::mpi_type_traits<uint32_t>::category, kamping::TypeCategory::integer);

    EXPECT_EQ(kamping::mpi_type_traits<float>::category, kamping::TypeCategory::floating);
    EXPECT_EQ(kamping::mpi_type_traits<double>::category, kamping::TypeCategory::floating);
    EXPECT_EQ(kamping::mpi_type_traits<long double>::category, kamping::TypeCategory::floating);

    EXPECT_EQ(kamping::mpi_type_traits<bool>::category, kamping::TypeCategory::logical);
    EXPECT_EQ(kamping::mpi_type_traits<kamping::kabool>::category, kamping::TypeCategory::logical);

    EXPECT_EQ(kamping::mpi_type_traits<std::complex<float>>::category, kamping::TypeCategory::complex);
    EXPECT_EQ(kamping::mpi_type_traits<std::complex<double>>::category, kamping::TypeCategory::complex);
    EXPECT_EQ(kamping::mpi_type_traits<std::complex<long double>>::category, kamping::TypeCategory::complex);

    EXPECT_EQ(kamping::mpi_type_traits<std::complex<int>>::category, kamping::TypeCategory::undefined);
    EXPECT_EQ(kamping::mpi_type_traits<char>::category, kamping::TypeCategory::character);
    EXPECT_EQ(kamping::mpi_type_traits<DummyType>::category, kamping::TypeCategory::kamping_provided);
}

TEST(MpiDataTypeTest, kabool_basics) {
    // size matches bool
    EXPECT_EQ(sizeof(kabool), sizeof(bool));
    // construction + explicit conversion
    EXPECT_EQ(static_cast<bool>(kabool{}), false);
    EXPECT_EQ(static_cast<bool>(kabool{false}), false);
    EXPECT_EQ(static_cast<bool>(kabool{true}), true);
    EXPECT_EQ(static_cast<kabool>(false), kabool{false});
    EXPECT_EQ(static_cast<kabool>(true), kabool{true});
    // implicit conversion
    EXPECT_EQ(kabool{false}, false);
    EXPECT_EQ(kabool{true}, true);
    EXPECT_EQ(kabool{true} && kabool{false}, false);
    EXPECT_EQ(kabool{true} && kabool{true}, true);
    EXPECT_EQ(kabool{false} || kabool{false}, false);
    EXPECT_EQ(kabool{true} || kabool{false}, true);
}

TEST(MpiDataTypeTest, register_types_with_environment) {
    int          c_array[3];
    MPI_Datatype array_type = mpi_datatype<decltype(c_array)>();

    struct TestStruct {
        int a;
        int b;
    };
    MPI_Datatype struct_type = mpi_datatype<TestStruct>();
    // should not register the type again
    MPI_Datatype other_struct_type = mpi_datatype<TestStruct>();
    (void)other_struct_type;

    freed_types.clear();
    mpi_env.free_registered_mpi_types();
    std::set<MPI_Datatype> expected_types({array_type, struct_type});
    EXPECT_EQ(freed_types, expected_types);
    freed_types.clear();
}
