#pragma once

#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"

namespace kamping::pipes {

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange>
auto make_vbuf(DataBuffer&& data, SizeRange&& size_v) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs();
}

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange, IntContiguousRange DisplsRange>
auto make_vbuf(DataBuffer&& data, SizeRange&& size_v, DisplsRange&& displs) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | with_displs(std::forward<DisplsRange>(displs));
}

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange>
auto make_vbuf_resizing(DataBuffer&& data, SizeRange&& size_v) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs() | resize_vbuf();
}

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange, IntContiguousRange DisplsRange>
auto make_vbuf_resizing(DataBuffer&& data, SizeRange&& size_v, DisplsRange&& displs) {
    return std::forward<DataBuffer>(data) | with_size_v(std::forward<SizeRange>(size_v)) | with_displs(std::forward<DisplsRange>(displs)) | resize_vbuf();
}


template <typename T>
auto make_vbuf_auto() {
    return std::vector<T>() | auto_size_v() | auto_displs() | resize_vbuf();
}

template <DataBufferConcept DataBuffer>
requires HasResize<DataBuffer>
auto make_vbuf_auto() {
    return DataBuffer{} | auto_size_v() | auto_displs() | resize_vbuf();
}

template <DataBufferConcept DataBuffer, IntContiguousRange SizeRange>
requires HasResize<DataBuffer>
auto make_vbuf_auto(SizeRange&& size_v) {
    return DataBuffer{} | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs() | resize_vbuf();
}

template <typename T, IntContiguousRange SizeRange>
auto make_vbuf_auto(SizeRange&& size_v) {
    return std::vector<T>{} | with_size_v(std::forward<SizeRange>(size_v)) | auto_displs() | resize_vbuf();
}

template <DataBufferConcept DataBuffer>
auto make_vbuf_auto(DataBuffer&& data) {
    return std::forward<DataBuffer>(data) | auto_size_v() | auto_displs();
};

template <DataBufferConcept DataBuffer>
auto make_vbuf_auto_resizing(DataBuffer&& data) {
    return std::forward<DataBuffer>(data) | auto_size_v() | auto_displs() | resize_vbuf();
};

} // namespace kamping::pipes
