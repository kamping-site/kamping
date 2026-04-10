#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <mpi.h>

namespace kamping::bridge {

template <typename T>
concept builtin_mpi_handle = std::same_as<T, MPI_Comm> || std::same_as<T, MPI_Datatype> || std::same_as<T, MPI_Request>
                             || std::same_as<T, MPI_Status> || std::same_as<T, MPI_Message>;

template <typename T>
concept ptr_to_builtin_mpi_handle =
    std::is_pointer_v<T> && !std::is_const_v<T> && builtin_mpi_handle<std::remove_pointer_t<T>>;

template <typename T>
concept has_mpi_native_handle_member = requires(T const& t) {
    { t.mpi_native_handle() } -> builtin_mpi_handle;
};

template <typename T>
concept has_mpi_native_handle_ptr_member = requires(T& t) {
    { t.mpi_native_handle_ptr() } -> ptr_to_builtin_mpi_handle;
};

// ──────────────────────────────────────────────────────────────────────────────
// native_handle_traits — trait class for non-intrusive MPI handle customization.
//
// Specialize native_handle_traits<T> for types you don't own. Only implement
// the members you need; unimplemented members fall back to the corresponding
// mpi_native_handle() / mpi_native_handle_ptr() member functions.
//
//   template <>
//   struct kamping::bridge::native_handle_traits<MyComm> {
//       static MPI_Comm handle(MyComm const& c)  { return c.get_comm(); }
//       static MPI_Comm* handle_ptr(MyComm& c)   { return c.get_comm_ptr(); }
//   };
// ──────────────────────────────────────────────────────────────────────────────
template <typename T>
struct native_handle_traits {};

template <typename T>
concept native_handle_traits_has_handle = requires(T const& t) {
    { native_handle_traits<T>::handle(t) } -> builtin_mpi_handle;
};

template <typename T>
concept native_handle_traits_has_handle_ptr = requires(T& t) {
    { native_handle_traits<T>::handle_ptr(t) } -> ptr_to_builtin_mpi_handle;
};

// ──────────────────────────────────────────────────────────────────────────────
// native_handle() dispatch — priority: native_handle_traits > member > builtin passthrough
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires native_handle_traits_has_handle<std::remove_cvref_t<T>>
constexpr auto native_handle(T const& t) {
    return native_handle_traits<std::remove_cvref_t<T>>::handle(t);
}

template <typename T>
    requires(!native_handle_traits_has_handle<std::remove_cvref_t<T>>) && has_mpi_native_handle_member<T>
constexpr auto native_handle(T const& t) {
    return t.mpi_native_handle();
}

template <builtin_mpi_handle T>
constexpr T native_handle(T t) noexcept {
    return t;
}

// ──────────────────────────────────────────────────────────────────────────────
// native_handle_ptr() dispatch — priority: native_handle_traits > member > builtin passthrough
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires native_handle_traits_has_handle_ptr<std::remove_cvref_t<T>>
constexpr auto native_handle_ptr(T& t) {
    return native_handle_traits<std::remove_cvref_t<T>>::handle_ptr(t);
}

template <typename T>
    requires(!native_handle_traits_has_handle_ptr<std::remove_cvref_t<T>>) && has_mpi_native_handle_ptr_member<T>
constexpr auto native_handle_ptr(T& t) {
    return t.mpi_native_handle_ptr();
}

template <builtin_mpi_handle T>
constexpr T* native_handle_ptr(T& t) noexcept {
    return &t;
}

template <builtin_mpi_handle T>
constexpr T* native_handle_ptr(T* t) noexcept {
    return t;
}

template <typename T, typename HandleType>
concept convertible_to_mpi_handle = builtin_mpi_handle<HandleType> && requires(T const& t) {
    { kamping::bridge::native_handle(t) } -> std::same_as<HandleType>;
};

template <typename T, typename HandleType>
concept convertible_to_mpi_handle_ptr = builtin_mpi_handle<HandleType> && requires(T& t) {
    { kamping::bridge::native_handle_ptr(t) } -> std::same_as<HandleType*>;
};

// ──────────────────────────────────────────────────────────────────────────────
// to_rank() / to_tag() — customization points for MPI rank and tag values.
//
// Dispatch priority:
//   1. rank_traits<T>::rank(t) / tag_traits<T>::tag(t)  — for types you don't own
//   2. Scoped enums — via std::to_underlying, then narrowing to int
//   3. Implicit int conversion — int, unscoped enums, etc.
//
// To support a typed rank wrapper or a scoped enum with non-int underlying type:
//
//   template <>
//   struct kamping::bridge::rank_traits<Rank> {
//       static int rank(Rank const& r) { return r.value; }
//   };
//
//   enum class Tag : uint8_t { DATA = 0, SYNC = 1 };
//   // No specialization needed — to_underlying handles it.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
struct rank_traits {};

template <typename T>
struct tag_traits {};

template <typename T>
concept rank_traits_has_rank = requires(T const& t) {
    { rank_traits<T>::rank(t) } -> std::same_as<int>;
};

template <typename T>
concept tag_traits_has_tag = requires(T const& t) {
    { tag_traits<T>::tag(t) } -> std::same_as<int>;
};

template <typename T>
    requires rank_traits_has_rank<std::remove_cvref_t<T>>
constexpr int to_rank(T const& t) {
    return rank_traits<std::remove_cvref_t<T>>::rank(t);
}

template <typename T>
    requires(!rank_traits_has_rank<std::remove_cvref_t<T>>) && std::is_scoped_enum_v<std::remove_cvref_t<T>>
constexpr int to_rank(T t) {
    return static_cast<int>(std::to_underlying(t));
}

template <typename T>
    requires(!rank_traits_has_rank<std::remove_cvref_t<T>>) && (!std::is_scoped_enum_v<std::remove_cvref_t<T>>)
            && std::convertible_to<T, int>
constexpr int to_rank(T t) {
    return static_cast<int>(t);
}

template <typename T>
    requires tag_traits_has_tag<std::remove_cvref_t<T>>
constexpr int to_tag(T const& t) {
    return tag_traits<std::remove_cvref_t<T>>::tag(t);
}

template <typename T>
    requires(!tag_traits_has_tag<std::remove_cvref_t<T>>) && std::is_scoped_enum_v<std::remove_cvref_t<T>>
constexpr int to_tag(T t) {
    return static_cast<int>(std::to_underlying(t));
}

template <typename T>
    requires(!tag_traits_has_tag<std::remove_cvref_t<T>>) && (!std::is_scoped_enum_v<std::remove_cvref_t<T>>)
            && std::convertible_to<T, int>
constexpr int to_tag(T t) {
    return static_cast<int>(t);
}

template <typename T>
concept mpi_rank = requires(T const& t) {
    { kamping::bridge::to_rank(t) } -> std::same_as<int>;
};

template <typename T>
concept mpi_tag = requires(T const& t) {
    { kamping::bridge::to_tag(t) } -> std::same_as<int>;
};

} // namespace kamping::bridge
