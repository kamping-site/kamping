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
#include "kamping/data_buffer.hpp"

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
    // typename T,
    typename OutArchive /*= cereal::BinaryOutputArchive*/,
    typename InArchive /*= cereal::BinaryInputArchive*/,
    typename Allocator /*= std::allocator<char> */,
    typename DataBufferType>
class SerializationBuffer {
private:
    std::basic_string<char, std::char_traits<char>, Allocator> _data; ///< Buffer holding the serialized data.
    DataBufferType                                             _object;

public:
    using data_type = typename DataBufferType::value_type;

    // template <
    //     bool enabled                    = deserialization_method == DeserializationMethod::Disabled,
    //     std::enable_if_t<enabled, bool> = true>
    SerializationBuffer(DataBufferType&& object) : _data(), _object(std::move(object)) {}

    // /// @brief Construct a serialization buffer from an object.
    // /// The object is serialized using the provided archive.
    // /// @tparam _T Type of the object to serialize. Default is \p T.
    // /// @tparam _OutArchive Type of the archive to use for serialization. Defaults the buffer's \p OutArchive.
    // /// @tparam Enable SFINAE to enable this constructor only if and \p OutArchive is provided.
    // // template <
    // //     bool enabled                    = deserialization_method == DeserializationMethod::ToSelf,
    // //     std::enable_if_t<enabled, bool> = true>
    // SerializationBuffer(T const& object) : _data(), Base(object) {}

    // // template <
    // //     bool enabled                    = deserialization_method == DeserializationMethod::ToSelf,
    // //     std::enable_if_t<enabled, bool> = true>
    // SerializationBuffer(T& object) : _data(), Base(object) {}

    // void serializef() {
    //     serialize(this->_object);
    // }

    template <typename _OutArchive = OutArchive, typename Enable = std::enable_if_t<!std::is_void_v<_OutArchive>>>
    void serialize() {
        std::basic_stringstream<char, std::char_traits<char>, Allocator> buffer;
        {
            OutArchive archive(buffer);
            archive(_object.underlying());
        }
        _data = buffer.str();
    }

    DataBufferType extract() && {
        return std::move(_object);
    }

    // void deserialize_self() {
    //     deserialize(this->_object);
    // }

    // /// @brief Deserialize the data in the buffer into an object of type \p _T.
    // /// @tparam _T Type of the object to deserialize into. Default is \p T.
    // /// @tparam _InArchive Type of the archive to use for deserialization. Defaults the buffer's \p InArchive.
    // /// @tparam Enable SFINAE to enable this function only if \p _InArchive are provided.
    // /// @return The deserialized object.
    // template <typename _InArchive = InArchive, typename Enable = std::enable_if_t<!std::is_void_v<_InArchive>>>
    // T deserialize() const {
    //     T result;
    //     deserialize(result);
    //     return result;
    // }

    /// @brief Deserialize the data in the buffer into an object of type \p _T. See \ref deserialize() for details.
    /// Instead of returning the deserialized object, the result is written into the provided reference.
    template <typename _InArchive = InArchive, typename Enable = std::enable_if_t<!std::is_void_v<_InArchive>>>
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

namespace internal {
/// @brief Tag type to identify serialization support.
struct serialization_support_tag {};

/// @brief Type trait to check if a type is a serialization buffer.
template <typename>
constexpr bool is_serialization_buffer_v_impl = false;

/// @brief Type trait to check if a type is a serialization buffer.
template <typename... Args>
constexpr bool is_serialization_buffer_v_impl<SerializationBuffer<Args...>> = true;

/// @brief Type trait to check if a type is a serialization buffer.
template <typename T>
constexpr bool is_serialization_buffer_v =
    is_serialization_buffer_v_impl<std::remove_const_t<std::remove_reference_t<T>>>;

template <bool serialization_used, typename BufferType>
auto deserialization_repack(BufferType buffer) {
    if constexpr (serialization_used) {
        auto serialization_data = buffer.extract();
        serialization_data.deserialize();
        return std::move(serialization_data).extract();
    } else {
        std::move(buffer);
    }
}
} // namespace internal

/// @brief Serializes an object using [`cereal`](https://uscilab.github.io/cereal/) and construct a buffer holding
/// the serialized data.
/// @tparam Archive Type of the archive to use for serialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryOutputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
/// @tparam T Type of the object to serialize.
template <typename Archive = cereal::BinaryOutputArchive, typename Allocator = std::allocator<char>, typename T>
auto as_serialized(T const& data) {
    internal::GenericDataBuffer<
        T,
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferOwnership::referencing,
        internal::BufferType::in_buffer>
        buffer(data);

    return SerializationBuffer<Archive, void, Allocator, decltype(buffer)>{std::move(buffer)};
}

template <
    typename OutArchive = cereal::BinaryOutputArchive,
    typename InArchive  = cereal::BinaryInputArchive,
    typename Allocator  = std::allocator<char>,
    typename T>
auto as_serialized(T&& data) {
    if constexpr (std::is_rvalue_reference_v<T&&>) {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType::send_recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::in_out_buffer>
            buffer(data);
        return SerializationBuffer<OutArchive, InArchive, Allocator, decltype(buffer)>{std::move(buffer)};
    } else {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType::send_recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::referencing,
            internal::BufferType::in_out_buffer>
            buffer(data);
        return SerializationBuffer<OutArchive, InArchive, Allocator, decltype(buffer)>{std::move(buffer)};
    }
}

/// @brief Constructs a buffer that can be used to deserialize an object using
/// [`cereal`](https://uscilab.github.io/cereal/).
/// @tparam T Type to deserialize into. This parameter is optional and can be omitted if the type deserilization
/// type is provided explicitly in the call to \ref SerializationBuffer::deserialize.
/// @tparam Archive Type of the archive to use for deserialization (see
/// https://uscilab.github.io/cereal/serialization_archives.html). Default is `cereal::BinaryInputArchive`.
/// @tparam Allocator Type of the allocator to use for the buffer holding the serialized data. Default is
/// `std::allocator<char>`.
template <typename T, typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>>
auto as_deserializable() {
    internal::GenericDataBuffer<
        T,
        internal::ParameterType::recv_buf,
        internal::BufferModifiability::modifiable,
        internal::BufferOwnership::owning,
        internal::BufferType::out_buffer>
        buffer{T{}};
    return SerializationBuffer<void, Archive, Allocator, decltype(buffer)>{std::move(buffer)};
}

template <typename Archive = cereal::BinaryInputArchive, typename Allocator = std::allocator<char>, typename T>
auto as_deserializable(T&& object) {
    if constexpr (std::is_rvalue_reference_v<T&&>) {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType::recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::out_buffer>
            buffer(std::move(object));
        return SerializationBuffer<void, Archive, Allocator, decltype(buffer)>(std::move(buffer));
    } else {
        internal::GenericDataBuffer<
            std::remove_reference_t<T>,
            internal::ParameterType::recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::referencing,
            internal::BufferType::out_buffer>
            buffer(object);
        return SerializationBuffer<void, Archive, Allocator, decltype(buffer)>(std::move(buffer));
    }
}

} // namespace kamping
