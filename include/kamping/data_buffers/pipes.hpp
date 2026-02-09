#pragma once

#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"

namespace kamping::pipes {

inline constexpr auto make_vbuf = []<typename DataBuffer, typename SizeRange>(
    DataBuffer && data, SizeRange&& size_v
) requires DataBufferConcept<DataBuffer> && IntContiguousRange<SizeRange> {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs()
           | resize_vbuf();
};

template <typename T>
inline constexpr auto make_vbuf_vector = []<typename SizeRange>(SizeRange&& size_v
) requires IntContiguousRange<SizeRange> {
    return std::vector<T>{} | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs() | resize_vbuf();
};

} // namespace kamping::pipes
