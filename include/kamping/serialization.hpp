#pragma once

#include <type_traits>

#include "cereal/archives/binary.hpp"

namespace kamping {

template <
    typename T,
    typename OutArchive = cereal::BinaryOutputArchive,
    typename InArchive  = cereal::BinaryInputArchive,
    typename Allocator  = std::allocator<char>>
struct SerializationBuffer {
    SerializationBuffer(std::basic_string<char, std::char_traits<char>, Allocator> data = {})
        : _data(std::move(data)) {}

    template <
        typename _T          = T,
        typename _OutArchive = OutArchive,
        typename Enable      = std::enable_if_t<!std::is_void_v<_OutArchive>>>
    SerializationBuffer(_T const& data) {
        std::basic_stringstream<char, std::char_traits<char>, Allocator> buffer;
        {
            OutArchive archive(buffer);
            archive(data);
        }
        _data = buffer.str();
    }

    std::basic_string<char, std::char_traits<char>, Allocator> _data;

    template <
        typename _T         = T,
        typename _InArchive = InArchive,
        typename Enable     = std::enable_if_t<!std::is_void_v<_InArchive>>>
    _T deserialize() const {
        static_assert(!std::is_void_v<_T>, "Please provide a type to deserialize into.");
        _T result;
        deserialize<_T, _InArchive>(result);
        return result;
    }

    template <
        typename _T         = T,
        typename _InArchive = InArchive,
        typename Enable     = std::enable_if_t<!std::is_void_v<_T> && !std::is_void_v<_InArchive>>>
    void deserialize(_T& result) const {
        static_assert(!std::is_void_v<_T>, "Please provide a type to deserialize into.");
        std::istringstream buffer(std::string(_data.begin(), _data.end()));
        {
            _InArchive archive(buffer);
            archive(result);
        }
    }

    using value_type = char;

    char* data() noexcept {
        return _data.data();
    }

    char const* data() const noexcept {
        return _data.data();
    }

    void resize(size_t size) {
        _data.resize(size);
    }

    size_t size() const {
        return _data.size();
    }
};

namespace internal {
struct serialization_support_tag {};

template <typename>
constexpr bool is_serialization_buffer_v_impl = false;

template <typename T, typename OutArchive, typename InArchive, typename Allocator>
constexpr bool is_serialization_buffer_v_impl<SerializationBuffer<T, OutArchive, InArchive, Allocator>> = true;

template <typename T>
constexpr bool is_serialization_buffer_v =
    is_serialization_buffer_v_impl<std::remove_const_t<std::remove_reference_t<T>>>;
} // namespace internal

template <typename T = void, typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>>
auto as_deserializable() {
    return SerializationBuffer<T, void, Archive, Allocator>();
}

template <typename T, typename Archive = cereal::BinaryOutputArchive, typename Allocator = std::allocator<char>>
auto as_serialized(T const& data) {
    std::basic_stringstream<char, std::char_traits<char>, Allocator> buffer;
    {
        Archive archive(buffer);
        archive(data);
    }
    return SerializationBuffer<T, Archive, void, Allocator>{buffer.str()};
}
} // namespace kamping
