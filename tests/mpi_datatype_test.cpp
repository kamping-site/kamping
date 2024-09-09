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

#include "helpers_for_testing.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_datatype.hpp"

using namespace ::kamping;
using namespace ::testing;

std::set<MPI_Datatype> freed_types;
size_t                 num_commit_calls = 0;

int MPI_Type_commit(MPI_Datatype* type) {
    num_commit_calls++;
    return PMPI_Type_commit(type);
}

int MPI_Type_free(MPI_Datatype* type) {
    auto it = freed_types.insert(*type);
    if (!it.second) {
        ADD_FAILURE() << "Type " << *type << " was freed twice";
    }
    return PMPI_Type_free(type);
}

MATCHER_P3(ResizedType, inner, lb, extend, "") {
    int num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(arg, &num_integers, &num_addresses, &num_datatypes, &combiner);
    if (combiner != MPI_COMBINER_RESIZED) {
        *result_listener << "not a resized type";
        return false;
    }
    MPI_Datatype            underlying_type;
    std::array<MPI_Aint, 2> type_bounds;
    MPI_Type_get_contents(
        arg,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &underlying_type
    );
    if (type_bounds[0] != static_cast<MPI_Aint>(lb)) {
        *result_listener << "wrong lb";
        return false;
    }
    if (type_bounds[1] != static_cast<MPI_Aint>(extend)) {
        *result_listener << "wrong extend";
        return false;
    }
    return ExplainMatchResult(inner, underlying_type, result_listener);
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
    if (count != static_cast<int>(n)) {
        *result_listener << "wrong count";
        return false;
    }
    return underlying_type == type;
}

class StructTypeMatcher {
public:
    StructTypeMatcher(std::initializer_list<MPI_Datatype> types) : expected_types(types) {}
    using is_gtest_matcher = void;
    std::vector<MPI_Datatype> expected_types;

    bool MatchAndExplain(MPI_Datatype type, MatchResultListener* listener) const {
        int num_integers, num_addresses, num_datatypes, combiner;
        MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
        if (combiner != MPI_COMBINER_STRUCT) {
            *listener << "is not a struct type";
            return false;
        }
        std::vector<int>          integers(static_cast<size_t>(num_integers));
        std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
        std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
        MPI_Type_get_contents(
            type,
            num_integers,
            num_addresses,
            num_datatypes,
            integers.data(),
            addresses.data(),
            datatypes.data()
        );
        size_t count = static_cast<size_t>(integers[0]);
        for (size_t i = 0; i < count; i++) {
            if (integers[i + 1] != 1) {
                *listener << "blocksize should be 1 for type " << i;
                return false;
            }
            if (datatypes[i] != expected_types[i]) {
                *listener << "type " << i << " does not match expected type";
                return false;
            }
        }
        return true;
    }
    void DescribeTo(std::ostream* os) const {
        *os << "is a struct type of the provided types";
    }
    void DescribeNegationTo(std::ostream* os) const {
        *os << "is not a struct type of the provided types";
    }
};
Matcher<MPI_Datatype> StructType(std::initializer_list<MPI_Datatype> types) {
    return StructTypeMatcher(types);
}

TEST(MpiDataTypeTest, mpi_datatype_basics) {
    // Check using the equivalent_mpi_datatypes() helper.
    EXPECT_THAT(possible_mpi_datatypes<char>(), Contains(mpi_type_traits<char>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned char>(), Contains(mpi_type_traits<unsigned char>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<signed char>(), Contains(mpi_type_traits<signed char>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_type_traits<uint8_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_type_traits<int8_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<wchar_t>(), Contains(mpi_type_traits<wchar_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<short>(), Contains(mpi_type_traits<short>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned short>(), Contains(mpi_type_traits<unsigned short>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<signed short>(), Contains(mpi_type_traits<signed short>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(mpi_type_traits<int>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned>(), Contains(mpi_type_traits<unsigned>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<long>(), Contains(mpi_type_traits<long>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<unsigned long>(), Contains(mpi_type_traits<unsigned long>::data_type()));
    EXPECT_THAT(
        possible_mpi_datatypes<signed long long int>(),
        Contains(mpi_type_traits<signed long long int>::data_type())
    );
    EXPECT_THAT(
        possible_mpi_datatypes<unsigned long long int>(),
        Contains(mpi_type_traits<unsigned long long int>::data_type())
    );
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(mpi_type_traits<float>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<double>(), Contains(mpi_type_traits<double>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<long double>(), Contains(mpi_type_traits<long double>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_type_traits<int8_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int16_t>(), Contains(mpi_type_traits<int16_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int32_t>(), Contains(mpi_type_traits<int32_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<int64_t>(), Contains(mpi_type_traits<int64_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_type_traits<uint8_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<uint16_t>(), Contains(mpi_type_traits<uint16_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<uint32_t>(), Contains(mpi_type_traits<uint32_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(mpi_type_traits<uint64_t>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<bool>(), Contains(mpi_type_traits<bool>::data_type()));
    EXPECT_THAT(possible_mpi_datatypes<kamping::kabool>(), Contains(mpi_type_traits<kamping::kabool>::data_type()));
    EXPECT_THAT(
        possible_mpi_datatypes<std::complex<double>>(),
        Contains(mpi_type_traits<std::complex<double>>::data_type())
    );
    EXPECT_THAT(
        possible_mpi_datatypes<std::complex<float>>(),
        Contains(mpi_type_traits<std::complex<float>>::data_type())
    );
    EXPECT_THAT(
        possible_mpi_datatypes<std::complex<long double>>(),
        Contains(mpi_type_traits<std::complex<long double>>::data_type())
    );
}

TEST(MpiDataTypeTest, mpi_datatype_const) {
    // Ignore const qualifiers.
    EXPECT_THAT(possible_mpi_datatypes<int8_t>(), Contains(mpi_type_traits<int8_t const>::data_type()));
}

TEST(MpiDataTypeTest, mpi_datatype_typedefs_and_using) {
    // typedefs and using directives.
    typedef int myInt;
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(mpi_type_traits<myInt>::data_type()));

    using myFloat = float;
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(mpi_type_traits<myFloat>::data_type()));
}

TEST(MpiDataTypeTest, mpi_datatype_size_t) {
    // size_t, which should be one of the unsigned integer types with at least 16 bits (as of C++11).
    EXPECT_THAT(
        (std::array{MPI_UNSIGNED_SHORT, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG}),
        Contains(mpi_type_traits<size_t>::data_type())
    );

    // As should std::size_t, which should be one of the unsigned integer types with at least 16 bits (as of C++11).
    EXPECT_THAT(
        (std::array{MPI_UNSIGNED_SHORT, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG}),
        Contains(mpi_type_traits<std::size_t>::data_type())
    );
}

TEST(MpiDataTypeTest, mpi_datatype_enum) {
    // Calling mpi_datatype with a enum type should use the underlying type.

    // Unscoped enum
    enum unscopedEnum { valueA = 0, valueB = 1 };
    auto unscopedEnum_types = possible_mpi_datatypes<std::underlying_type_t<unscopedEnum>>();
    EXPECT_THAT(unscopedEnum_types, Contains(mpi_type_traits<unscopedEnum>::data_type()));

    // Unscoped enum with explicit underlying type
    enum unscopedEnumInt : int { valueA2 = 0, valueB2 = 1 };
    auto unscopedEnumInt_types = possible_mpi_datatypes<std::underlying_type_t<unscopedEnumInt>>();
    EXPECT_THAT(unscopedEnumInt_types, Contains(mpi_type_traits<unscopedEnumInt>::data_type()));

    // Scoped enum
    enum class scopedEnum { valueA = 0, valueB = 1 };
    auto scopedEnum_types = possible_mpi_datatypes<std::underlying_type_t<scopedEnum>>();
    EXPECT_THAT(scopedEnum_types, Contains(mpi_type_traits<scopedEnum>::data_type()));

    // Scope enum with explicit underlying type
    enum class scopedEnumUint8_t : uint8_t { valueA = 0, valueB = 1 };
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(mpi_type_traits<scopedEnumUint8_t>::data_type()));

    enum class scopedEnumInt64_t : int64_t { valueA = 0, valueB = 1 };
    EXPECT_THAT(possible_mpi_datatypes<int64_t>(), Contains(mpi_type_traits<scopedEnumInt64_t>::data_type()));
}

TEST(MpiDataTypeTest, contiguous_type_works) {
    std::array<float, 3> a               = {1.0f, 2.0f, 3.0f};
    MPI_Datatype         contiguous_type = kamping::contiguous_type<float, 3>::data_type();
    int                  num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(contiguous_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_CONTIGUOUS);
    // returned values for MPI_COMBINER_CONTIGUOUS according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 1);
    EXPECT_EQ(num_addresses, 0);
    EXPECT_EQ(num_datatypes, 1);
    int          count;
    MPI_Datatype underlying_type;
    MPI_Type_get_contents(
        contiguous_type,
        num_integers,
        num_addresses,
        num_datatypes,
        &count,
        nullptr,
        &underlying_type
    );
    EXPECT_EQ(count, 3);
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(MPI_FLOAT));
    // now pack our array into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&contiguous_type);
    int pack_size;
    MPI_Pack_size(1, contiguous_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char> buffer(static_cast<size_t>(pack_size));
    int               position = 0;
    MPI_Pack(a.data(), 1, contiguous_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    std::array<float, 3> b;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, b.data(), 1, contiguous_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_THAT(b, ElementsAreArray(a));
    PMPI_Type_free(&contiguous_type);
}

TEST(MpiDataTypeTest, byte_serialized_type_works) {
    std::pair<int, double> a                    = {1, 2.0};
    MPI_Datatype           byte_serialized_type = kamping::byte_serialized<std::pair<int, double>>::data_type();
    int                    num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(byte_serialized_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_CONTIGUOUS);
    // returned values for MPI_COMBINER_CONTIGUOUS
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 1);
    EXPECT_EQ(num_addresses, 0);
    EXPECT_EQ(num_datatypes, 1);
    int          count;
    MPI_Datatype underlying_type;
    MPI_Type_get_contents(
        byte_serialized_type,
        num_integers,
        num_addresses,
        num_datatypes,
        &count,
        nullptr,
        &underlying_type
    );
    EXPECT_EQ(count, sizeof(a));
    EXPECT_EQ(underlying_type, MPI_BYTE);
    // now pack our type into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&byte_serialized_type);
    int pack_size;
    MPI_Pack_size(1, byte_serialized_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char> buffer(static_cast<size_t>(pack_size));
    int               position = 0;
    MPI_Pack(&a, 1, byte_serialized_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    std::pair<int, double> b;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &b, 1, byte_serialized_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_EQ(b, a);
    PMPI_Type_free(&byte_serialized_type);
}

#ifdef KAMPING_ENABLE_REFLECTION
TEST(MpiDataTypeTest, struct_type_works_with_struct) {
    struct TestStruct {
        uint8_t  a;
        uint64_t b;
    };
    MPI_Datatype resized_type = kamping::struct_type<TestStruct>::data_type();
    int          num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(resized_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_RESIZED);
    // returned values for MPI_COMBINER_RESIZED
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 0);
    EXPECT_EQ(num_addresses, 2);
    EXPECT_EQ(num_datatypes, 1);
    std::array<MPI_Aint, 2> type_bounds;
    MPI_Datatype            struct_type;
    MPI_Type_get_contents(
        resized_type,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &struct_type
    );
    EXPECT_EQ(type_bounds[0], 0);                  // lb
    EXPECT_EQ(type_bounds[1], sizeof(TestStruct)); // extent
    MPI_Type_get_envelope(struct_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_STRUCT);
    // returned values for MPI_COMBINER_STRUCT
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 3);  // count + 1
    EXPECT_EQ(num_addresses, 2); // count
    EXPECT_EQ(num_datatypes, 2); // count
    std::vector<int>          integers(static_cast<size_t>(num_integers));
    std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
    std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
    MPI_Type_get_contents(
        struct_type,
        num_integers,
        num_addresses,
        num_datatypes,
        integers.data(),
        addresses.data(),
        datatypes.data()
    );
    EXPECT_EQ(integers[0], 2);                                               // i[0] == count
    EXPECT_EQ(integers[1], 1);                                               // i[1] == blocklength[0]
    EXPECT_EQ(integers[2], 1);                                               // i[2] == blocklength[1]
    EXPECT_EQ(addresses[0], offsetof(TestStruct, a));                        // a[0] == displacements[0]
    EXPECT_EQ(addresses[1], offsetof(TestStruct, b));                        // a[1] == displacements[1]
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(datatypes[0]));  // d[0] == types[0]
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(datatypes[1])); // d[1] == types[1]
    // now pack our struct into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&struct_type);
    int pack_size;
    MPI_Pack_size(1, struct_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char> buffer(static_cast<size_t>(pack_size));
    int               position = 0;
    TestStruct        t        = {1, 2};
    MPI_Pack(&t, 1, struct_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    TestStruct u;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &u, 1, struct_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_EQ(u.a, t.a);
    EXPECT_EQ(u.b, t.b);
    PMPI_Type_free(&struct_type);
}

struct ExplicitNestedStruct {
    float c;
    bool  d;
};
struct ImplicitNestedStruct {
    float c;
    bool  d;
};
namespace kamping {
template <>
struct mpi_type_traits<ExplicitNestedStruct> : struct_type<ExplicitNestedStruct> {};
} // namespace kamping

TEST(MpiDataTypeTest, struct_type_works_with_nested_struct) {
    struct TestStruct {
        uint8_t              a;
        uint64_t             b;
        ExplicitNestedStruct nested;          // should use the explicit struct MPI type declaration
        ImplicitNestedStruct implicit_nested; // should use byte serialized type
    };
    MPI_Datatype resized_type = kamping::struct_type<TestStruct>::data_type();
    int          num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(resized_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_RESIZED);
    // returned values for MPI_COMBINER_RESIZED
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 0);
    EXPECT_EQ(num_addresses, 2);
    EXPECT_EQ(num_datatypes, 1);
    std::array<MPI_Aint, 2> type_bounds;
    MPI_Datatype            struct_type;
    MPI_Type_get_contents(
        resized_type,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &struct_type
    );
    EXPECT_EQ(type_bounds[0], 0);                  // lb
    EXPECT_EQ(type_bounds[1], sizeof(TestStruct)); // extent
    MPI_Type_get_envelope(struct_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_STRUCT);
    // returned values for MPI_COMBINER_STRUCT
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 5);  // count + 1
    EXPECT_EQ(num_addresses, 4); // count
    EXPECT_EQ(num_datatypes, 4); // count
    std::vector<int>          integers(static_cast<size_t>(num_integers));
    std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
    std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
    MPI_Type_get_contents(
        struct_type,
        num_integers,
        num_addresses,
        num_datatypes,
        integers.data(),
        addresses.data(),
        datatypes.data()
    );
    EXPECT_EQ(integers[0], 4);                                               // i[0] == count
    EXPECT_EQ(integers[1], 1);                                               // i[1] == blocklength[0]
    EXPECT_EQ(integers[2], 1);                                               // i[2] == blocklength[1]
    EXPECT_EQ(integers[3], 1);                                               // i[3] == blocklength[2]
    EXPECT_EQ(integers[4], 1);                                               // i[4] == blocklength[3]
    EXPECT_EQ(addresses[0], offsetof(TestStruct, a));                        // a[0] == displacements[0]
    EXPECT_EQ(addresses[1], offsetof(TestStruct, b));                        // a[1] == displacements[1]
    EXPECT_EQ(addresses[2], offsetof(TestStruct, nested));                   // a[2] == displacements[2]
    EXPECT_EQ(addresses[3], offsetof(TestStruct, implicit_nested));          // a[3] == displacements[3]
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(datatypes[0]));  // d[0] == types[0]
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(datatypes[1])); // d[1] == types[1]

    MPI_Datatype explicit_nested_type = datatypes[2];
    MPI_Datatype implicit_nested_type = datatypes[3];
    MPI_Type_get_envelope(explicit_nested_type, &num_integers, &num_addresses, &num_datatypes, &combiner);

    EXPECT_EQ(combiner, MPI_COMBINER_RESIZED);
    // returned values for MPI_COMBINER_RESIZED
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 0);
    EXPECT_EQ(num_addresses, 2);
    EXPECT_EQ(num_datatypes, 1);
    MPI_Datatype explicit_nested_type_inner_struct;
    MPI_Type_get_contents(
        explicit_nested_type,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &explicit_nested_type_inner_struct
    );
    EXPECT_EQ(type_bounds[0], 0);                            // lb
    EXPECT_EQ(type_bounds[1], sizeof(ExplicitNestedStruct)); // extent
    MPI_Type_get_envelope(explicit_nested_type_inner_struct, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_STRUCT);
    // returned values for MPI_COMBINER_STRUCT
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 3);  // count + 1
    EXPECT_EQ(num_addresses, 2); // count
    EXPECT_EQ(num_datatypes, 2); // count
    integers.resize(static_cast<size_t>(num_integers));
    addresses.resize(static_cast<size_t>(num_addresses));
    datatypes.resize(static_cast<size_t>(num_datatypes));
    MPI_Type_get_contents(
        explicit_nested_type_inner_struct,
        num_integers,
        num_addresses,
        num_datatypes,
        integers.data(),
        addresses.data(),
        datatypes.data()
    );
    EXPECT_EQ(integers[0], 2);                                            // i[0] == count
    EXPECT_EQ(integers[1], 1);                                            // i[1] == blocklength[0]
    EXPECT_EQ(integers[2], 1);                                            // i[2] == blocklength[1]
    EXPECT_EQ(addresses[0], offsetof(ExplicitNestedStruct, c));           // a[0] == displacements[0]
    EXPECT_EQ(addresses[1], offsetof(ExplicitNestedStruct, d));           // a[1] == displacements[1]
    EXPECT_THAT(possible_mpi_datatypes<float>(), Contains(datatypes[0])); // d[0] == types[0]
    EXPECT_THAT(possible_mpi_datatypes<bool>(), Contains(datatypes[1]));  // d[1] == types[1]

    MPI_Type_get_envelope(implicit_nested_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_CONTIGUOUS);
    // returned values for MPI_COMBINER_CONTIGUOUS
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 1);
    EXPECT_EQ(num_addresses, 0);
    EXPECT_EQ(num_datatypes, 1);
    integers.resize(static_cast<size_t>(num_integers));
    addresses.resize(static_cast<size_t>(num_addresses));
    datatypes.resize(static_cast<size_t>(num_datatypes));
    int          count;
    MPI_Datatype underlying_type;
    MPI_Type_get_contents(
        implicit_nested_type,
        num_integers,
        num_addresses,
        num_datatypes,
        &count,
        nullptr,
        &underlying_type
    );
    EXPECT_EQ(count, sizeof(ImplicitNestedStruct));
    EXPECT_EQ(underlying_type, MPI_BYTE);

    // now pack our struct into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&struct_type);
    int pack_size;
    MPI_Pack_size(1, struct_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char> buffer(static_cast<size_t>(pack_size));
    int               position = 0;
    TestStruct        t        = {1, 2, {3.0f, true}, {4.0f, false}};
    MPI_Pack(&t, 1, struct_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    TestStruct u;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &u, 1, struct_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_EQ(u.a, t.a);
    EXPECT_EQ(u.b, t.b);
    EXPECT_EQ(u.nested.c, t.nested.c);
    EXPECT_EQ(u.nested.d, t.nested.d);
    PMPI_Type_free(&struct_type);
}
#endif

TEST(MpiDataTypeTest, struct_type_works_with_pair) {
    MPI_Datatype resized_type = kamping::struct_type<std::pair<uint8_t, uint64_t>>::data_type();
    int          num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(resized_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_RESIZED);
    // returned values for MPI_COMBINER_RESIZED
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 0);
    EXPECT_EQ(num_addresses, 2);
    EXPECT_EQ(num_datatypes, 1);
    std::array<MPI_Aint, 2> type_bounds;
    MPI_Datatype            struct_type;
    MPI_Type_get_contents(
        resized_type,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &struct_type
    );
    EXPECT_EQ(type_bounds[0], 0);                                    // lb
    EXPECT_EQ(type_bounds[1], sizeof(std::pair<uint8_t, uint64_t>)); // extent
    MPI_Type_get_envelope(struct_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_STRUCT);
    // returned values for MPI_COMBINER_STRUCT
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 3);  // count + 1
    EXPECT_EQ(num_addresses, 2); // count
    EXPECT_EQ(num_datatypes, 2); // count
    std::vector<int>          integers(static_cast<size_t>(num_integers));
    std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
    std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
    MPI_Type_get_contents(
        struct_type,
        num_integers,
        num_addresses,
        num_datatypes,
        integers.data(),
        addresses.data(),
        datatypes.data()
    );
    EXPECT_EQ(integers[0], 2); // i[0] == count
    EXPECT_EQ(integers[1], 1); // i[1] == blocklength[0]
    EXPECT_EQ(integers[2], 1); // i[2] == blocklength[1]
    using pair_type = std::pair<uint8_t, uint64_t>;
    EXPECT_EQ(addresses[0], offsetof(pair_type, first));                     // a[0] == displacements[0]
    EXPECT_EQ(addresses[1], offsetof(pair_type, second));                    // a[1] == displacements[1]
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(datatypes[0]));  // d[0] == types[0]
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(datatypes[1])); // d[1] == types[1]
    // now pack our pair into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&struct_type);
    int pack_size;
    MPI_Pack_size(1, struct_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char>            buffer(static_cast<size_t>(pack_size));
    int                          position = 0;
    std::pair<uint8_t, uint64_t> t        = {1, 2};
    MPI_Pack(&t, 1, struct_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    std::pair<uint8_t, uint64_t> u;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &u, 1, struct_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_EQ(u, t);
    PMPI_Type_free(&struct_type);
}

TEST(MpiDataTypeTest, struct_type_works_with_tuple) {
    using Tuple               = std::tuple<uint8_t, uint64_t>;
    MPI_Datatype resized_type = kamping::struct_type<Tuple>::data_type();
    int          num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(resized_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_RESIZED);
    // returned values for MPI_COMBINER_RESIZED
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 0);
    EXPECT_EQ(num_addresses, 2);
    EXPECT_EQ(num_datatypes, 1);
    std::array<MPI_Aint, 2> type_bounds;
    MPI_Datatype            struct_type;
    MPI_Type_get_contents(
        resized_type,
        num_integers,
        num_addresses,
        num_datatypes,
        nullptr,
        type_bounds.data(),
        &struct_type
    );
    EXPECT_EQ(type_bounds[0], 0);             // lb
    EXPECT_EQ(type_bounds[1], sizeof(Tuple)); // extent
    MPI_Type_get_envelope(struct_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    EXPECT_EQ(combiner, MPI_COMBINER_STRUCT);
    // returned values for MPI_COMBINER_STRUCT
    // according to section 5.1.13 of the MPI standard (Decoding a Datatype)
    EXPECT_EQ(num_integers, 3);  // count + 1
    EXPECT_EQ(num_addresses, 2); // count
    EXPECT_EQ(num_datatypes, 2); // count
    std::vector<int>          integers(static_cast<size_t>(num_integers));
    std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
    std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
    MPI_Type_get_contents(
        struct_type,
        num_integers,
        num_addresses,
        num_datatypes,
        integers.data(),
        addresses.data(),
        datatypes.data()
    );
    EXPECT_EQ(integers[0], 2); // i[0] == count
    EXPECT_EQ(integers[1], 1); // i[1] == blocklength[0]
    EXPECT_EQ(integers[2], 1); // i[2] == blocklength[1]
    Tuple    tuple;
    MPI_Aint base_address = reinterpret_cast<MPI_Aint>(&tuple);
    EXPECT_EQ(addresses[0], reinterpret_cast<MPI_Aint>(&std::get<0>(tuple)) - base_address); // a[0] == displacements[0]
    EXPECT_EQ(addresses[1], reinterpret_cast<MPI_Aint>(&std::get<1>(tuple)) - base_address); // a[1] == displacements[1]
    EXPECT_THAT(possible_mpi_datatypes<uint8_t>(), Contains(datatypes[0]));                  // d[0] == types[0]
    EXPECT_THAT(possible_mpi_datatypes<uint64_t>(), Contains(datatypes[1]));                 // d[1] == types[1]
    // now pack our tuple into a buffer and and unpack it again to check if the datatype works
    MPI_Type_commit(&struct_type);
    int pack_size;
    MPI_Pack_size(1, struct_type, MPI_COMM_WORLD, &pack_size);
    std::vector<char>             buffer(static_cast<size_t>(pack_size));
    int                           position = 0;
    std::tuple<uint8_t, uint64_t> t        = {1, 2};
    MPI_Pack(&t, 1, struct_type, buffer.data(), static_cast<int>(buffer.size()), &position, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    position = 0;
    std::tuple<uint8_t, uint64_t> u;
    MPI_Unpack(buffer.data(), static_cast<int>(buffer.size()), &position, &u, 1, struct_type, MPI_COMM_WORLD);
    EXPECT_EQ(position, pack_size);
    EXPECT_EQ(u, t);
    PMPI_Type_free(&struct_type);
}

template <typename T1, typename T2>
struct TestStruct {
    T1 a;
    T2 b;
};
struct Empty {};

namespace kamping {
template <>
struct mpi_type_traits<std::tuple<int, double, std::complex<float>>>
    : struct_type<std::tuple<int, double, std::complex<float>>> {};
template <>
struct mpi_type_traits<std::pair<int, double>> : byte_serialized<std::pair<int, double>> {};
} // namespace kamping

TEST(MpiDataTypeTest, mpi_datatype_struct) {
    EXPECT_THAT(
        (mpi_type_traits<TestStruct<int, int>>::data_type()),
        ContiguousType(MPI_BYTE, sizeof(TestStruct<int, int>))
    );
    EXPECT_EQ((mpi_type_traits<TestStruct<int, int>>::category), TypeCategory::contiguous);

    EXPECT_THAT(
        (mpi_type_traits<TestStruct<double, int>>::data_type()),
        ContiguousType(MPI_BYTE, sizeof(TestStruct<double, int>))
    );
    EXPECT_EQ((mpi_type_traits<TestStruct<double, int>>::category), TypeCategory::contiguous);

    EXPECT_THAT(
        (mpi_type_traits<TestStruct<int, double>>::data_type()),
        ContiguousType(MPI_BYTE, sizeof(TestStruct<int, double>))
    );
    EXPECT_EQ((mpi_type_traits<TestStruct<int, double>>::category), TypeCategory::contiguous);

    EXPECT_THAT(
        (mpi_type_traits<TestStruct<int, Empty>>::data_type()),
        ContiguousType(MPI_BYTE, sizeof(TestStruct<int, Empty>))
    );
    EXPECT_EQ((mpi_type_traits<TestStruct<int, Empty>>::category), TypeCategory::contiguous);

    // pair is not trivially copyable, but we defined a byte_serialized trait for it explicitly.
    EXPECT_THAT(
        (mpi_type_traits<std::pair<int, double>>::data_type()),
        ContiguousType(MPI_BYTE, sizeof(std::pair<int, double>))
    );
    EXPECT_EQ((mpi_type_traits<std::pair<int, double>>::category), TypeCategory::contiguous);

    EXPECT_THAT(
        (mpi_type_traits<std::tuple<int, double, std::complex<float>>>::data_type()),
        ResizedType(
            StructType({MPI_INT, MPI_DOUBLE, MPI_CXX_FLOAT_COMPLEX}),
            0,
            sizeof(std::tuple<int, double, std::complex<float>>)
        )
    );

    // struct is no trivially copyable, but we defined a struct_type trait for it explicitly.
    EXPECT_EQ((mpi_type_traits<std::tuple<int, double, std::complex<float>>>::category), TypeCategory::struct_like);
    EXPECT_THAT(
        (mpi_type_traits<std::tuple<int, double, std::complex<float>>>::data_type()),
        ResizedType(
            StructType({MPI_INT, MPI_DOUBLE, MPI_CXX_FLOAT_COMPLEX}),
            0,
            sizeof(std::tuple<int, double, std::complex<float>>)
        )
    );
}

TEST(MpiDataTypeTest, mpi_datatype_c_array) {
    // Calling mpi_datatype with an array should return a continuous datatype.
    {
        int c_array[3];
        EXPECT_THAT(mpi_type_traits<decltype(c_array)>::data_type(), ContiguousType(MPI_INT, 3));
        EXPECT_EQ(mpi_type_traits<decltype(c_array)>::category, TypeCategory::contiguous);
    }
    {
        std::array<int, 3> cpp_array;
        EXPECT_THAT(mpi_type_traits<decltype(cpp_array)>::data_type(), ContiguousType(MPI_INT, 3));
        EXPECT_EQ(mpi_type_traits<decltype(cpp_array)>::category, TypeCategory::contiguous);
    }

    {
        std::array<double, 3> cpp_array;
        EXPECT_THAT(mpi_type_traits<decltype(cpp_array)>::data_type(), ContiguousType(MPI_DOUBLE, 3));
        EXPECT_EQ(mpi_type_traits<decltype(cpp_array)>::category, TypeCategory::contiguous);
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

    EXPECT_EQ(kamping::mpi_type_traits<std::complex<int>>::category, kamping::TypeCategory::contiguous);
    EXPECT_EQ(kamping::mpi_type_traits<char>::category, kamping::TypeCategory::character);
    EXPECT_EQ(kamping::mpi_type_traits<DummyType>::category, kamping::TypeCategory::contiguous);
}

TEST(MpiDataTypeTest, has_static_type_test) {
    struct DummyType {
        int  a;
        char b;
    };
    EXPECT_TRUE(has_static_type_v<int>);
    EXPECT_TRUE(has_static_type_v<DummyType>);
    EXPECT_FALSE((has_static_type_v<std::pair<int, int>>));
    EXPECT_FALSE((has_static_type_v<std::tuple<int, int>>));
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
    // Setup
    mpi_env.free_registered_mpi_types();
    freed_types.clear();
    num_commit_calls = 0;

    int          c_array[3];
    MPI_Datatype array_type = mpi_datatype<decltype(c_array)>();
    EXPECT_EQ(num_commit_calls, 1);

    struct TestStruct {
        int a;
        int b;
    };
    MPI_Datatype struct_type = mpi_datatype<TestStruct>();
    EXPECT_EQ(num_commit_calls, 2);
    // should not register the type again
    MPI_Datatype other_struct_type = mpi_datatype<TestStruct>();
    EXPECT_EQ(num_commit_calls, 2);

    // just do something with the types to avoid optimizing it away
    int size;
    MPI_Pack_size(1, other_struct_type, MPI_COMM_WORLD, &size);

    freed_types.clear();
    mpi_env.free_registered_mpi_types();
    std::set<MPI_Datatype> expected_types({array_type, struct_type});
    EXPECT_EQ(freed_types, expected_types);
    freed_types.clear();
}
