#pragma once

#include <ranges>
#include <vector>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/mpi_datatype.hpp"

namespace kamping {

struct range_resizable_tag {};
struct size_v_resizable_tag {};

// Concepts for the Data Buffers
template <typename Buff>
concept Typed = requires(Buff buf) {
    { type(buf) } -> std::same_as<MPI_Datatype>;
};

template <typename Buff>
concept DataBufferConcept = std::ranges::contiguous_range<Buff> && std::ranges::sized_range<Buff> && Typed<Buff>;

template <typename SBuff>
concept SendDataBuffer = DataBufferConcept<SBuff> && std::ranges::input_range<SBuff>;

template <typename RBuff>
concept RecvDataBuffer =
    DataBufferConcept<RBuff> && std::ranges::output_range<RBuff, std::ranges::range_value_t<RBuff>>;

template <typename T>
concept IntContiguousRange = std::ranges::contiguous_range<T> && std::same_as < std::ranges::range_value_t<T>,
int > &&std::ranges::sized_range<T>;

template <typename T>
concept RefToIntContiguousRange = std::is_reference_v<T> && IntContiguousRange<std::remove_reference_t<T>>;

template <typename Buff>
concept HasSizeV = requires(Buff buf) {
    { buf.size_v() } -> RefToIntContiguousRange;
};

template <typename Buff>
concept HasDispls = requires(Buff buf) {
    { buf.displs() } -> RefToIntContiguousRange;
};

template <typename Buff>
concept ExtendedDataBuffer = DataBufferConcept<Buff> && HasDispls<Buff> && HasSizeV<Buff>;

template<typename T>
concept HasResize = requires(T buf, size_t size) {
    {buf.resize(size)};
};

// Concepts for infer tags

template <typename Buff>
concept ResizableBuffer = requires {typename Buff::infer_tag;}
        && std::same_as<typename Buff::infer_tag, range_resizable_tag>
        && HasResize<Buff>;

template <typename Buff>
concept ResizableSizeV = requires {typename Buff::infer_tag;}
        && std::same_as<typename Buff::infer_tag, size_v_resizable_tag>
        && HasResize<decltype(std::declval<Buff>().size_v())>;

template <typename Buff>
concept HasInferTag = requires {typename Buff::infer_tag;};

} // namespace kamping
