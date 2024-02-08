#pragma once
#include <complex>
#include <type_traits>

#include <mpi.h>

#include "kamping/kabool.hpp"

namespace kamping {
/// @brief the members specify which group the datatype belongs to according to the type groups specified in
/// Section 6.9.2 of the MPI 4.0 standard.
enum class TypeCategory { integer, floating, complex, logical, byte, character, struct_like, contiguous };
constexpr bool category_has_to_be_committed(TypeCategory category) {
    switch (category) {
        case TypeCategory::integer:
        case TypeCategory::floating:
        case TypeCategory::complex:
        case TypeCategory::logical:
        case TypeCategory::byte:
        case TypeCategory::character:
            return false;
        case TypeCategory::struct_like:
        case TypeCategory::contiguous:
            return true;
    }
}

template <typename T>
struct builtin_type : std::false_type {};

template <typename T>
constexpr bool is_builtin_type_v = builtin_type<T>::value;

template <>
struct builtin_type<char> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::character;
};

template <>
struct builtin_type<signed char> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_SIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<unsigned char> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<wchar_t> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_WCHAR;
    }
    static constexpr TypeCategory category = TypeCategory::character;
};

template <>
struct builtin_type<short int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<unsigned short int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_INT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<unsigned int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<long int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<unsigned long int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<long long int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<unsigned long long int> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct builtin_type<float> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_FLOAT;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct builtin_type<double> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct builtin_type<long double> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_LONG_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct builtin_type<bool> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct builtin_type<kabool> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct builtin_type<std::complex<float>> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CXX_FLOAT_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};
template <>
struct builtin_type<std::complex<double>> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

template <>
struct builtin_type<std::complex<long double>> : std::true_type {
    static MPI_Datatype data_type() {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

} // namespace kamping
