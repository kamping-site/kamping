#pragma once
#include <complex>

#include "./kabool.hpp"
#include "./traits.hpp"

namespace kamping {

template <>
struct mpi_type_traits<char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CHAR;
    }
};

template <>
struct mpi_type_traits<signed char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_SIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<unsigned char> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_CHAR;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<wchar_t> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_WCHAR;
    }
};

template <>
struct mpi_type_traits<short int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<unsigned short int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_SHORT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_INT;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<unsigned int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<unsigned long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<unsigned long long int> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_UNSIGNED_LONG_LONG;
    }
    static constexpr TypeCategory category = TypeCategory::integer;
};

template <>
struct mpi_type_traits<float> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_FLOAT;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits<double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits<long double> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_LONG_DOUBLE;
    }
    static constexpr TypeCategory category = TypeCategory::floating;
};

template <>
struct mpi_type_traits<bool> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct mpi_type_traits<kabool> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_BOOL;
    }
    static constexpr TypeCategory category = TypeCategory::logical;
};

template <>
struct mpi_type_traits<std::complex<float>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_FLOAT_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};
template <>
struct mpi_type_traits<std::complex<double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

template <>
struct mpi_type_traits<std::complex<long double>> : is_builtin_mpi_type_true {
    static MPI_Datatype data_type() {
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    }
    static constexpr TypeCategory category = TypeCategory::complex;
};

} // namespace kamping
