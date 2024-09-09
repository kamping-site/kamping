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

#ifdef KAMPING_ENABLE_SERIALIZATION
    #include "cereal/archives/binary.hpp"
#endif
#include "kamping/data_buffer.hpp"

namespace kamping {
namespace internal {
#ifdef KAMPING_ENABLE_SERIALIZATION

/// @brief Buffer holding serialized data.
///
/// This uses [`cereal`](https://uscilab.github.io/cereal/) to serialize and deserialize objects.
///
/// @tparam OutArchive Type of the archive to use for serialization.
/// @tparam InArchive Type of the archive to use for deserialization.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data.
/// @tparam DataBufferType Type of the \ref GenericDataBuffer holding the data to serialize/deserialize into.
template <typename OutArchive, typename InArchive, typename Allocator, typename DataBufferType>
class SerializationBuffer {
private:
    std::basic_string<char, std::char_traits<char>, Allocator> _data; ///< Buffer holding the serialized data.
    DataBufferType _object; ///< Object to de/serialize encapsulated in a \ref GenericDataBuffer.

public:
    using data_type =
        typename DataBufferType::value_type; ///< Type of the encapsulated object to serialize/deserialize.

    /// @brief Construct a serialization buffer from a \ref GenericDataBuffer containing the object to
    /// serialize/deserialize into.
    SerializationBuffer(DataBufferType&& object) : _data(), _object(std::move(object)) {}

    /// @brief Serialize the object into the character buffer stored internally.
    void serialize() {
        std::basic_stringstream<char, std::char_traits<char>, Allocator> buffer;
        {
            OutArchive archive(buffer);
            archive(_object.underlying());
        }
        _data = buffer.str();
    }

    /// @brief Extract the \ref GenericDataBuffer containing the encapsulated object.
    DataBufferType extract() && {
        return std::move(_object);
    }

    /// @brief Deserialize from the character buffer stored internally into the encapsulated object.
    void deserialize() {
        std::istringstream buffer(std::string(_data.begin(), _data.end()));
        {
            InArchive archive(buffer);
            archive(_object.underlying());
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
#endif

/// @brief Tag type to identify serialization support.
struct serialization_support_tag {};

/// @brief Type trait to check if a type is a serialization buffer.
template <typename>
constexpr bool is_serialization_buffer_v_impl = false;
#ifdef KAMPING_ENABLE_SERIALIZATION
/// @brief Type trait to check if a type is a serialization buffer.
template <typename... Args>
constexpr bool is_serialization_buffer_v_impl<SerializationBuffer<Args...>> = true;
#endif

/// @brief Type trait to check if a type is a serialization buffer.
template <typename T>
constexpr bool is_serialization_buffer_v =
    is_serialization_buffer_v_impl<std::remove_const_t<std::remove_reference_t<T>>>;

/// @brief If \p serialization_used is true, this takes a received serialization buffer, deserializes the data and
/// repacks it into a new buffer only containing the deserialized data
/// If \p serialization_used is false, the input buffer is returned unchanged.
template <bool serialization_used, typename BufferType>
auto deserialization_repack(BufferType buffer) {
    if constexpr (serialization_used) {
        auto serialization_data = buffer.extract();
        serialization_data.deserialize();
        return std::move(serialization_data).extract();
    } else {
        return buffer;
    }
}
} // namespace internal
#ifdef KAMPING_ENABLE_SERIALIZATION
/// @brief Serializes an object using [`cereal`](https://uscilab.github.io/cereal/).
/// @tparam Archive Type of the archive to use for serialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryOutputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
/// @tparam T Type of the object to serialize.
template <typename Archive = cereal::BinaryOutputArchive, typename Allocator = std::allocator<char>, typename T>
auto as_serialized(T const& data) {
    internal::GenericDataBuffer<
        T,
        internal::ParameterType,
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferOwnership::referencing,
        internal::BufferType::in_buffer>
        buffer(data);

    return internal::SerializationBuffer<Archive, void, Allocator, decltype(buffer)>{std::move(buffer)};
}

/// @brief Serializes and deserializes an object using [`cereal`](https://uscilab.github.io/cereal/).
/// If the input object is an rvalue reference, the result of deserialization is returned by the surrounding
/// communication call. If the input object is an lvalue reference, the input object is modified in place.
/// @tparam OutArchive Type of the archive to use for serialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryOutputArchive`.
/// @tparam InArchive Type of the archive to use for deserialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
/// @tparam T Type of the object to serialize.
template <
    typename OutArchive = cereal::BinaryOutputArchive,
    typename InArchive  = cereal::BinaryInputArchive,
    typename Allocator  = std::allocator<char>,
    typename T>
auto as_serialized(T&& data) {
    if constexpr (std::is_rvalue_reference_v<T&&>) {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType,
            internal::ParameterType::send_recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::in_out_buffer>
            buffer(data);
        return internal::SerializationBuffer<OutArchive, InArchive, Allocator, decltype(buffer)>{std::move(buffer)};
    } else {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType,
            internal::ParameterType::send_recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::referencing,
            internal::BufferType::in_out_buffer>
            buffer(data);
        return internal::SerializationBuffer<OutArchive, InArchive, Allocator, decltype(buffer)>{std::move(buffer)};
    }
}

/// @brief Deserializes the received data using [`cereal`](https://uscilab.github.io/cereal/) and returns it in the
/// result of the surrounding communication call.
/// @tparam T Type to deserialize into.
/// @tparam Archive Type of the archive to use for deserialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
template <typename T, typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>>
auto as_deserializable() {
    internal::GenericDataBuffer<
        T,
        internal::ParameterType,
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferOwnership::owning,
        internal::BufferType::out_buffer>
        buffer{T{}};
    return internal::SerializationBuffer<void, Archive, Allocator, decltype(buffer)>{std::move(buffer)};
}

/// @brief Deserializes the received data using [`cereal`](https://uscilab.github.io/cereal/) into the input object.
/// If the input object is an rvalue reference, the result of deserialization is returned by the surrounding
/// communication call. If the input object is an lvalue reference, the input object is modified in place.
/// @tparam Archive Type of the archive to use for deserialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
/// @tparam T Type to deserialize into.
template <typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>, typename T>
auto as_deserializable(T&& object) {
    if constexpr (std::is_rvalue_reference_v<T&&>) {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType,
            internal::ParameterType::recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::out_buffer>
            buffer(std::move(object));
        return internal::SerializationBuffer<void, Archive, Allocator, decltype(buffer)>(std::move(buffer));
    } else {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType,
            internal::ParameterType::recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::referencing,
            internal::BufferType::out_buffer>
            buffer(object);
        return internal::SerializationBuffer<void, Archive, Allocator, decltype(buffer)>(std::move(buffer));
    }
}
#endif

} // namespace kamping
