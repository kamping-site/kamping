#pragma once
#include <mpi.h>
#include <pfr.hpp>

#include "./traits.hpp"

namespace kamping {
/// @brief MPI type traits for std::array.
template <typename T, size_t N>
struct mpi_type_traits<std::array<T, N>> : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;
    static MPI_Datatype           data_type() {
                  static_assert(has_static_type<T>, "Array must have a static type");
                  static_assert(N > 0, "Array must have at least one element");
                  MPI_Datatype type;
                  MPI_Type_contiguous(static_cast<int>(N), mpi_type_traits<T>::data_type(), &type);
                  return type;
    }
};

/// @brief MPI type traits for plain C arrays.
template <typename ArrayType>
struct mpi_type_traits<ArrayType, std::enable_if_t<std::is_array_v<ArrayType>>>
    : mpi_type_traits<std::array<std::remove_extent_t<ArrayType>, std::extent_v<ArrayType>>> {};
} // namespace kamping

template <typename T, size_t N>
struct pfr::is_reflectable<std::array<T, N>, kamping::kamping_tag> : std::false_type {};

template <typename ArrayType>
struct pfr::is_reflectable<ArrayType, std::enable_if_t<std::is_array_v<ArrayType>, kamping::kamping_tag>>
    : std::false_type {};

namespace kamping {

template <typename T, typename Enable = void>
struct mpi_type_struct : is_builtin_mpi_type_false {
    static constexpr TypeCategory category    = TypeCategory::undefined;
    static MPI_Datatype           data_type() = delete;
};

/// @brief MPI type traits for std::pair.
template <typename T1, typename T2>
struct mpi_type_struct<std::pair<T1, T2>, std::enable_if_t<has_static_type<T1> && has_static_type<T2>>>
    : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;
    static MPI_Datatype           data_type() {
                  std::pair<T1, T2> t{};
                  MPI_Datatype      types[2]     = {mpi_type_traits<T1>::data_type(), mpi_type_traits<T2>::data_type()};
                  int               blocklens[2] = {1, 1};
                  MPI_Aint          base;
                  MPI_Get_address(&t, &base);
                  MPI_Aint disp[2];
                  MPI_Get_address(&t.first, &disp[0]);
                  MPI_Get_address(&t.second, &disp[1]);
                  disp[0] = MPI_Aint_diff(disp[0], base);
                  disp[1] = MPI_Aint_diff(disp[1], base);
                  MPI_Datatype type;
                  MPI_Type_create_struct(2, blocklens, disp, types, &type);
                  return type;
    }
};
} // namespace kamping
template <typename T1, typename T2>
struct pfr::is_reflectable<std::pair<T1, T2>, kamping::kamping_tag> : std::false_type {};

namespace kamping {
/// @brief MPI type traits for std::tuple.
template <typename... Ts>
struct mpi_type_struct<std::tuple<Ts...>, std::enable_if_t<all_have_static_types<Ts...>>> : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;

    static MPI_Datatype data_type() {
        std::tuple<Ts...>     t{};
        constexpr std::size_t tuple_size = sizeof...(Ts);

        MPI_Datatype types[tuple_size] = {mpi_type_traits<Ts>::data_type()...};
        int          blocklens[tuple_size];
        MPI_Aint     disp[tuple_size];
        MPI_Aint     base;
        MPI_Get_address(&t, &base);

        // Calculate displacements for each tuple element using std::apply and fold expressions
        size_t i = 0;
        std::apply(
            [&](auto&... elem) {
                (
                    [&] {
                        MPI_Get_address(&elem, &disp[i]);
                        disp[i]      = MPI_Aint_diff(disp[i], base);
                        blocklens[i] = 1;
                        i++;
                    }(),
                    ...
                );
            },
            t
        );

        MPI_Datatype type;
        MPI_Type_create_struct(tuple_size, blocklens, disp, types, &type);
        return type;
    }
};
} // namespace kamping

template <typename... Ts>
struct pfr::is_reflectable<std::tuple<Ts...>, kamping::kamping_tag> : std::false_type {};

namespace kamping {

/// @brief MPI type traits for enums.
template <typename E>
struct mpi_type_traits<
    E,
    std::enable_if_t<std::is_enum_v<E> && has_static_type<std::underlying_type_t<E>> && !std::is_array_v<E>>>
    : mpi_type_traits<std::underlying_type_t<E>> {};
} // namespace kamping

template <typename E>
struct pfr::is_reflectable<E, std::enable_if_t<std::is_enum_v<E>, kamping::kamping_tag>> : std::false_type {};

namespace kamping {
template <typename T, typename Enable = void>
struct reflectable {
    using member_types = std::tuple<>;
};

template <typename T>
struct reflectable<T, std::enable_if_t<pfr::is_implicitly_reflectable_v<T, kamping_tag>>> {
    using member_types = decltype(pfr::structure_to_tuple(std::declval<T>()));
};

template <typename T>
struct mpi_type_struct<
    T,
    std::enable_if_t<
        pfr::is_implicitly_reflectable<T, kamping_tag>::value
        && tuple_all_have_static_types<typename reflectable<T>::member_types>>>

    : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;

    static MPI_Datatype data_type() {
        T                     t{};
        constexpr std::size_t tuple_size = pfr::tuple_size_v<T>;

        MPI_Datatype types[tuple_size];
        int          blocklens[tuple_size];
        MPI_Aint     disp[tuple_size];
        MPI_Aint     base;
        MPI_Get_address(&t, &base);

        // Calculate displacements for each tuple element using std::apply and fold expressions
        pfr::for_each_field(t, [&](auto& elem, size_t i) {
            MPI_Get_address(&elem, &disp[i]);
            types[i]     = mpi_type_traits<std::remove_reference_t<decltype(elem)>>::data_type();
            disp[i]      = MPI_Aint_diff(disp[i], base);
            blocklens[i] = 1;
        });

        MPI_Datatype type;
        MPI_Type_create_struct(tuple_size, blocklens, disp, types, &type);
        return type;
    }
};

template <typename T>
struct mpi_type_contiguous_byte : is_builtin_mpi_type_false {
    static constexpr TypeCategory category = TypeCategory::kamping_provided;

    static MPI_Datatype data_type() {
        MPI_Datatype type;
        MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
        return type;
    }
};

} // namespace kamping
