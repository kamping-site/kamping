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

#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"

using namespace ::kamping;
using namespace ::testing;

// Returns a std::vector containing all MPI_Datatypes equivalent to the given C++ datatype on this machine.
// Removes the topmost level of const and volatile qualifiers.
template <typename T>
std::vector<MPI_Datatype> possible_mpi_datatypes() noexcept {
    // Remove const and volatile qualifiers.
    using T_no_cv = std::remove_cv_t<T>;

    // Check if we got a array type -> create a continuous type.
    if constexpr (std::is_array_v<T_no_cv>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        return std::vector<MPI_Datatype>{mpi_custom_continuous_type<sizeof(T_no_cv)>()};
    }

    // Check if we got a enum type -> use underlying type
    if constexpr (std::is_enum_v<T_no_cv>) {
        return possible_mpi_datatypes<std::underlying_type_t<T_no_cv>>();
    }

    // For each supported C++ datatype, check if it is equivalent to the T_no_cv and if so, add the corresponding MPI
    // datatype to the list of possible types.
    std::vector<MPI_Datatype> possible_mpi_datatypes;
    if constexpr (std::is_same_v<T_no_cv, char>) {
        possible_mpi_datatypes.push_back(MPI_CHAR);
    }
    if constexpr (std::is_same_v<T_no_cv, signed char>) {
        possible_mpi_datatypes.push_back(MPI_SIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_cv, unsigned char>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_cv, wchar_t>) {
        possible_mpi_datatypes.push_back(MPI_WCHAR);
    }
    if constexpr (std::is_same_v<T_no_cv, signed short>) {
        possible_mpi_datatypes.push_back(MPI_SHORT);
    }
    if constexpr (std::is_same_v<T_no_cv, unsigned short>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_SHORT);
    }
    if constexpr (std::is_same_v<T_no_cv, signed int>) {
        possible_mpi_datatypes.push_back(MPI_INT);
    }
    if constexpr (std::is_same_v<T_no_cv, unsigned int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED);
    }
    if constexpr (std::is_same_v<T_no_cv, signed long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG);
    }
    if constexpr (std::is_same_v<T_no_cv, unsigned long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG);
    }
    if constexpr (std::is_same_v<T_no_cv, signed long long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_cv, unsigned long long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_cv, float>) {
        possible_mpi_datatypes.push_back(MPI_FLOAT);
    }
    if constexpr (std::is_same_v<T_no_cv, double>) {
        possible_mpi_datatypes.push_back(MPI_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_cv, long double>) {
        possible_mpi_datatypes.push_back(MPI_LONG_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_cv, int8_t>) {
        possible_mpi_datatypes.push_back(MPI_INT8_T);
    }
    if constexpr (std::is_same_v<T_no_cv, int16_t>) {
        possible_mpi_datatypes.push_back(MPI_INT16_T);
    }
    if constexpr (std::is_same_v<T_no_cv, int32_t>) {
        possible_mpi_datatypes.push_back(MPI_INT32_T);
    }
    if constexpr (std::is_same_v<T_no_cv, int64_t>) {
        possible_mpi_datatypes.push_back(MPI_INT64_T);
    }
    if constexpr (std::is_same_v<T_no_cv, uint8_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT8_T);
    }
    if constexpr (std::is_same_v<T_no_cv, uint16_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT16_T);
    }
    if constexpr (std::is_same_v<T_no_cv, uint32_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT32_T);
    }
    if constexpr (std::is_same_v<T_no_cv, uint64_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT64_T);
    }
    if constexpr (std::is_same_v<T_no_cv, bool>) {
        possible_mpi_datatypes.push_back(MPI_C_BOOL);
    }
    if constexpr (std::is_same_v<T_no_cv, std::complex<float>>) {
        possible_mpi_datatypes.push_back(MPI_C_FLOAT_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_cv, std::complex<double>>) {
        possible_mpi_datatypes.push_back(MPI_C_DOUBLE_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_cv, std::complex<long double>>) {
        possible_mpi_datatypes.push_back(MPI_C_LONG_DOUBLE_COMPLEX);
    }

    // If not other type matched, this is a custom datatype.
    if (possible_mpi_datatypes.size() == 0) {
        possible_mpi_datatypes.push_back(mpi_custom_continuous_type<sizeof(T)>());
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
    EXPECT_THAT(possible_mpi_datatypes<std::complex<double>>(), Contains(mpi_datatype<std::complex<double>>()));
    EXPECT_THAT(possible_mpi_datatypes<std::complex<float>>(), Contains(mpi_datatype<std::complex<float>>()));
    EXPECT_THAT(
        possible_mpi_datatypes<std::complex<long double>>(), Contains(mpi_datatype<std::complex<long double>>()));
}

TEST(MpiDataTypeTest, mpi_datatype_const_and_volatile) {
    // Ignore const and volatile qualifiers.
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<const int8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<volatile int8_t>()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_datatype<const volatile int8_t>()));
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
        Contains(mpi_datatype<size_t>()));

    // As should std::size_t, which should be one of the unsigned integer types with at least 16 bits (as of C++11).
    EXPECT_THAT(
        (std::array{MPI_UNSIGNED_SHORT, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG}),
        Contains(mpi_datatype<std::size_t>()));
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
    EXPECT_NE(MPI_C_BOOL, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_C_FLOAT_COMPLEX, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_C_DOUBLE_COMPLEX, mpi_datatype<TestStruct>());
    EXPECT_NE(MPI_C_LONG_DOUBLE_COMPLEX, mpi_datatype<TestStruct>());
    EXPECT_EQ(mpi_datatype_size(mpi_datatype<TestStruct>()), 2 * sizeof(int));
}

TEST(MpiDataTypeTest, mpi_datatype_c_array) {
    // Calling mpi_datatype with an array should return a continuous datatype.
    int c_array[3];

    // There seems to be no way to check if a given datatype in MPI is a custom type, we therefore rule out that it's
    // equal to any of the other types, including the NULL type.
    EXPECT_NE(MPI_DATATYPE_NULL, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_CHAR, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_CHAR, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_SIGNED_CHAR, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UNSIGNED_CHAR, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_WCHAR, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_SHORT, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UNSIGNED_SHORT, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_INT, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UNSIGNED, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_LONG, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UNSIGNED_LONG, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_LONG_LONG, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UNSIGNED_LONG_LONG, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_FLOAT, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_DOUBLE, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_LONG_DOUBLE, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_INT8_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_INT16_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_INT32_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_INT64_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UINT8_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UINT16_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UINT32_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_UINT64_T, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_C_BOOL, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_C_FLOAT_COMPLEX, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_C_DOUBLE_COMPLEX, mpi_datatype<decltype(c_array)>());
    EXPECT_NE(MPI_C_LONG_DOUBLE_COMPLEX, mpi_datatype<decltype(c_array)>());
    EXPECT_EQ(mpi_datatype_size(mpi_datatype<decltype(c_array)>()), 3 * sizeof(int));
}

TEST(MpiDataTypeTest, mpi_datatype_size) {
    EXPECT_EQ(mpi_datatype_size(MPI_INT), sizeof(int));
    EXPECT_EQ(mpi_datatype_size(MPI_CHAR), sizeof(char));
    EXPECT_EQ(mpi_datatype_size(MPI_INT16_T), 2);

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Datatype null_type = MPI_DATATYPE_NULL;
    EXPECT_THROW(mpi_datatype_size(null_type), kamping::MpiErrorException);

    bool has_thrown = false;
    try {
        mpi_datatype_size(null_type);
    } catch (kamping::MpiErrorException& e) {
        has_thrown = true;
        EXPECT_EQ(e.mpi_error_code(), MPI_ERR_TYPE);
        EXPECT_THAT(e.what(), HasSubstr("Failed with the following error message:"));
        EXPECT_THAT(e.what(), HasSubstr("MPI_Type_size failed"));
    }
    EXPECT_TRUE(has_thrown);
}
