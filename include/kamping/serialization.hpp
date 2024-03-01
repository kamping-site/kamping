// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <type_traits>

#include "cereal/archives/binary.hpp"

namespace kamping {

/// @brief Buffer holding serialized data.
///
/// This uses [`cereal`](https://uscilab.github.io/cereal/) to serialize and deserialize objects.
///
/// @tparam T Type of the object to serialize.
/// @tparam OutArchive Type of the archive to use for serialization. Default is `cereal::BinaryOutputArchive`.
/// @tparam InArchive Type of the archive to use for deserialization. Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data.
template <
    typename T,
    typename OutArchive = cereal::BinaryOutputArchive,
    typename InArchive  = cereal::BinaryInputArchive,
    typename Allocator  = std::allocator<char>>
class SerializationBuffer {
private:
    std::basic_string<char, std::char_traits<char>, Allocator> _data; ///< Buffer holding the serialized data.

public:
    /// @brief Construct a serialization buffer from a string (a.k.a a buffer of chars). If no data is provided, the
    /// buffer is empty.
    SerializationBuffer(std::basic_string<char, std::char_traits<char>, Allocator> data = {})
        : _data(std::move(data)) {}

    /// @brief Construct a serialization buffer from an object.
    /// The object is serialized using the provided archive.
    /// @tparam _T Type of the object to serialize. Default is \p T.
    /// @tparam _OutArchive Type of the archive to use for serialization. Defaults the buffer's \p OutArchive.
    /// @tparam Enable SFINAE to enable this constructor only if and \p OutArchive is provided.
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

    /// @brief Deserialize the data in the buffer into an object of type \p _T.
    /// @tparam _T Type of the object to deserialize into. Default is \p T.
    /// @tparam _InArchive Type of the archive to use for deserialization. Defaults the buffer's \p InArchive.
    /// @tparam Enable SFINAE to enable this function only if \p _InArchive are provided.
    /// @return The deserialized object.
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

    /// @brief Deserialize the data in the buffer into an object of type \p _T. See \ref deserialize() for details.
    /// Instead of returning the deserialized object, the result is written into the provided reference.
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

    using value_type = char; ///< Type of the elements in the buffer.

    /// @brief Access the underlying buffer.
    char* data() noexcept {
        return _data.data();
    }

    /// @brief Access the underlying buffer.
    char const* data() const noexcept {
        return _data.data();
    }

    /// @brief Resize the underlying buffer.
    /// @param size New size of the buffer.
    void resize(size_t size) {
        _data.resize(size);
    }

    /// @brief Access the size of the underlying buffer.
    size_t size() const {
        return _data.size();
    }
};

namespace internal {
/// @brief Tag type to identify serialization support.
struct serialization_support_tag {};

/// @brief Type trait to check if a type is a serialization buffer.
template <typename>
constexpr bool is_serialization_buffer_v_impl = false;

/// @brief Type trait to check if a type is a serialization buffer.
template <typename T, typename OutArchive, typename InArchive, typename Allocator>
constexpr bool is_serialization_buffer_v_impl<SerializationBuffer<T, OutArchive, InArchive, Allocator>> = true;

/// @brief Type trait to check if a type is a serialization buffer.
template <typename T>
constexpr bool is_serialization_buffer_v =
    is_serialization_buffer_v_impl<std::remove_const_t<std::remove_reference_t<T>>>;
} // namespace internal

/// @brief Serializes an object using [`cereal`](https://uscilab.github.io/cereal/) and construct a buffer holding the
/// serialized data.
/// @tparam Archive Type of the archive to use for serialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryOutputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
/// @tparam T Type of the object to serialize.
template <typename Archive = cereal::BinaryOutputArchive, typename Allocator = std::allocator<char>, typename T>
auto as_serialized(T const& data) {
    return SerializationBuffer<T, Archive, void, Allocator>{data};
}

/// @brief Constructs a buffer that can be used to deserialize an object using
/// [`cereal`](https://uscilab.github.io/cereal/).
/// @tparam T Type to deserialize into. This parameter is optional and can be omitted if the type deserilization type is
/// provided explicitly in the call to \ref SerializationBuffer::deserialize.
/// @tparam Archive Type of the archive to use for deserialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
template <typename T = void, typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>>
auto as_deserializable() {
    return SerializationBuffer<T, void, Archive, Allocator>();
}

} // namespace kamping
