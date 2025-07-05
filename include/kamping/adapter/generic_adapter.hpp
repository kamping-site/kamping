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

template <typename T, typename Container, typename DataBufferType>
class AdapterBuffer {

public:
    using GetDataType = std::function<const T*(const Container&)>;
    using GetSizeType = std::function<std::size_t(const Container&)>;
    using value_type = T;

    explicit AdapterBuffer(DataBufferType&& object, GetDataType&& get_data, GetSizeType&& get_size) :
    _object(std::move(object)), _data(_object.underlying()), _get_data(std::move(get_data)), _get_size(std::move(get_size)) {}

    T const* data() const noexcept {
        return _get_data(_data);
    }

    [[nodiscard]] size_t size() const {
        return _get_size(_data);
    }

private:
    DataBufferType _object;
    const Container& _data;
    GetDataType _get_data;
    GetSizeType _get_size;

};

    template<typename T, typename Container>
    auto generic_adapter(const Container& data,
        std::function<const T*(const Container&)> get_data,
        std::function<size_t(const Container& data)> get_size ) {
        internal::GenericDataBuffer<
            Container,
            internal::ParameterType,
            internal::ParameterType::send_buf,
            internal::BufferModifiability::constant,
            internal::BufferOwnership::referencing,
            internal::BufferType::in_buffer>
        buffer(data);
        return AdapterBuffer<T, Container, decltype(buffer)>(std::move(buffer), std::move(get_data), std::move(get_size));
    }
}



