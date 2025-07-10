// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
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
#include <functional>

#include "kamping/data_buffer.hpp"

namespace kamping::adapter {

template <typename T, typename Container, typename DataBufferType, typename GetData, typename GetSize>
class AdapterBuffer {
public:
    using value_type = T;

    explicit AdapterBuffer(DataBufferType&& object, GetData&& get_data, GetSize&& get_size)
        : _object(std::move(object)),
          _data(_object.underlying()),
          _get_data(std::move(get_data)),
          _get_size(std::move(get_size)) {}

    T const* data() const noexcept {
        return _get_data(_data);
    }

    [[nodiscard]] size_t size() const {
        return _get_size(_data);
    }

private:
    DataBufferType   _object;
    Container const& _data;
    GetData          _get_data;
    GetSize          _get_size;
};

template <typename GetData, typename T, typename Container>
concept has_get_data = requires(GetData data_func, Container const& container) {
    { data_func(container) } -> std::convertible_to<const T*>;
};

template <typename GetSize, typename Container>
concept has_get_size = requires(GetSize size_func, Container const& container) {
    { size_func(container) } -> std::same_as<size_t>;
};

template <typename T, typename Container, typename GetData, typename GetSize>
requires has_get_data<GetData, T, Container> && has_get_size<GetSize, Container>
auto generic_adapter(Container const& data, GetData data_func, GetSize size_func) {
    internal::GenericDataBuffer<
        Container,
        internal::ParameterType,
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferOwnership::referencing,
        internal::BufferType::in_buffer>
        buffer(data);
    return AdapterBuffer<T, Container, decltype(buffer), GetData, GetSize>(
        std::move(buffer),
        std::move(data_func),
        std::move(size_func)
    );
}

template <typename T, typename Container, typename GetData, typename GetSize>
requires std::same_as<std::invoke_result_t<GetSize, Container const&>, size_t> && std::convertible_to
    < std::invoke_result_t<GetData, Container const&>,
const T* > auto generic_adapter_alt_helper(Container const& data, GetData data_func, GetSize size_func) {
    internal::GenericDataBuffer<
        Container,
        internal::ParameterType,
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferOwnership::referencing,
        internal::BufferType::in_buffer>
        buffer(data);
    return AdapterBuffer<T, Container, decltype(buffer), GetData, GetSize>(
        std::move(buffer),
        std::move(data_func),
        std::move(size_func)
    );
}

template <typename Container, typename GetData, typename GetSize>
auto generic_adapter_alt(Container const& data, GetData data_func, GetSize size_func) {
    using T = std::remove_pointer_t<std::invoke_result_t<GetData, Container const&>>;
    return generic_adapter_alt_helper<T>(data, data_func, size_func);
}

template <typename T, typename Container>
auto generic_adapter_std_func(
    Container const&                             data,
    std::function<const T*(Container const&)>    get_data,
    std::function<size_t(Container const& data)> get_size
) {
    internal::GenericDataBuffer<
        Container,
        internal::ParameterType,
        internal::ParameterType::send_buf,
        internal::BufferModifiability::constant,
        internal::BufferOwnership::referencing,
        internal::BufferType::in_buffer>
        buffer(data);
    return AdapterBuffer<T, Container, decltype(buffer), decltype(get_data), decltype(get_size)>(
        std::move(buffer),
        std::move(get_data),
        std::move(get_size)
    );
}

} // namespace kamping::adapter
