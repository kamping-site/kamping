#pragma once

#include <ranges>
#include <vector>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/mpi_datatype.hpp"

namespace kamping {

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

template <typename Buff>
concept HasSizeV = requires(Buff buf) {
    { buf.size_v() } -> IntContiguousRange<>;
};

template <typename Buff>
concept HasDisplacements = requires(Buff buf) {
    { buf.displacements() } -> IntContiguousRange<>;
};

template <typename Buff>
concept ExtendedDataBuffer = DataBufferConcept<Buff> && HasDisplacements<Buff> && HasSizeV<Buff>;

template <typename Buff>
concept HasSetSize = requires(Buff buf, size_t size) {
    {buf.set_size(size)};
};

template <typename Buff>
concept HasSetSizeV = requires(Buff buf, std::vector<int>&& sizes) {
    {buf.set_size_v(std::move(sizes))};
};

template <typename Buff>
concept HasSetDisplacements = requires(Buff buf, std::vector<int>&& displacements) {
    {buf.set_displacements(std::move(displacements))};
};

} // namespace kamping
