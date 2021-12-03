#include <mpi.h>

#include <cassert>
#include <type_traits>

namespace kamping {
///
/// @brief Creates a custom continuous MPI datatype.
///
/// @tparam NumBytes The number of bytes for the new type.
/// @return The newly created MPI_Datatype.
/// @see mpi_datatype()
///
template <size_t NumBytes>
[[nodiscard]] MPI_Datatype mpi_custom_continuous_type() noexcept {
    static_assert(NumBytes > 0, "You cannot create a continuous type with 0 bytes.");
    // Create a new MPI datatype only the first type per NumBytes this function is called.
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(NumBytes, MPI_CHAR, &type);
        MPI_Type_commit(&type);
        assert(type != MPI_DATATYPE_NULL);
    }
    // From the second call onwards, re-use the existing type.
    return type;
}

/// @brief Translate template parameter T to an MPI_Datatype. If no corresponding MPI_Datatype exists, we will create
/// new custom continuous type.
///        Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
/// @tparam T The type to translate into a MPI_Datatype.
/// @return The tag identifying the corresponding MPI_Datatype or the newly created type.
/// @see mpi_custom_continuous_type()
///
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_datatype() noexcept {
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

    if constexpr (std::is_same_v<T, char>) {
        mpi_type = MPI_CHAR;
    } else if constexpr (std::is_same_v<T, signed char>) {
        mpi_type = MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        mpi_type = MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, wchar_t>) {
        mpi_type = MPI_WCHAR;
    } else if constexpr (std::is_same_v<T, signed short>) {
        mpi_type = MPI_SHORT;
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        mpi_type = MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<T, signed int>) {
        mpi_type = MPI_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        mpi_type = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, signed long int>) {
        mpi_type = MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long int>) {
        mpi_type = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, signed long long int>) {
        mpi_type = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long long int>) {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        mpi_type = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        mpi_type = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        mpi_type = MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        mpi_type = MPI_INT8_T;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        mpi_type = MPI_INT16_T;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        mpi_type = MPI_INT32_T;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        mpi_type = MPI_INT64_T;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        mpi_type = MPI_UINT8_T;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        mpi_type = MPI_UINT16_T;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        mpi_type = MPI_UINT32_T;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        mpi_type = MPI_UINT64_T;
    } else if constexpr (std::is_same_v<T, bool>) {
        mpi_type = MPI_C_BOOL;
    } else if constexpr (std::is_same_v<T, float _Complex>) {
        mpi_type = MPI_C_FLOAT_COMPLEX;
    } else if constexpr (std::is_same_v<T, double _Complex>) {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same_v<T, long double _Complex>) {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    } else {
        mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    }

    assert(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}
} // namespace kamping
