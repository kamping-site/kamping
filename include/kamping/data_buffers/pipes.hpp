#pragma once

#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"

namespace kamping::pipes {

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange>
auto make_vbuf(DataBuffer&& data, SizeRange&& size_v) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs();
};

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange>
auto make_vbuf_resize(DataBuffer&& data, SizeRange&& size_v) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs()
           | resize_vbuf();
};

template <typename T, IntContiguousRange SizeRange>
auto make_vbuf_vector(SizeRange&& size_v) {
    return std::vector<T>{} | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs() | resize_vbuf();
};

template <typename T>
auto make_vbuf_auto(size_t size) {
    return std::vector<T>{} | with_size_v(std::vector<int>(size)) | auto_displs() | resize_vbuf();
};

} // namespace kamping::pipes
