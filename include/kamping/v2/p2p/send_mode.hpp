#pragma once

#include <concepts>
#include <type_traits>

namespace kamping::v2 {
namespace send_mode {
struct standard_t {};
struct buffered_t {};
struct sync_t {};
struct ready_t {};
constexpr standard_t standard{};
constexpr buffered_t buffered{};
constexpr sync_t     sync{};
constexpr ready_t    ready{};
} // namespace send_mode
  
template <typename T>
concept is_send_mode = std::same_as<std::remove_cvref_t<T>, send_mode::standard_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::buffered_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::sync_t>
                       || std::same_as<std::remove_cvref_t<T>, send_mode::ready_t>;

} // namespace kamping::v2
