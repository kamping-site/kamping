#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "../sample-sort/kamping.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kassert/kassert.hpp"

template <typename IndexType>
struct IRR {
    IndexType index;
    IndexType rank1;
    IndexType rank2;

    IRR(IndexType idx, IndexType r1, IndexType r2) : index(idx), rank1(r1), rank2(r2) {}
}; // struct IRR

template <typename IndexType>
auto reduce_alphabet(std::vector<std::uint8_t>&& input, kamping::BasicCommunicator& comm) {
    using InputType = std::remove_reference_t<decltype(input)>::value_type;

    std::array<IndexType, std::numeric_limits<InputType>::max() + 1> hist = {0};

    std::for_each(input.begin(), input.end(), [&hist](auto const symbol) { ++hist[symbol]; });

    comm.allreduce_inplace(kamping::send_recv_buf(hist), kamping::op(kamping::ops::plus<>()));

    size_t new_alphabet_size = 1;
    std::transform(hist.cbegin(), hist.cend(), hist.begin(), [&new_alphabet_size](auto const count) {
        return count > 0 ? new_alphabet_size++ : 0;
    });

    size_t const bits_per_symbol = std::ceil(std::log2(new_alphabet_size));

    size_t const k_fitting = (CHAR_BIT * sizeof(IndexType)) / bits_per_symbol;

    KASSERT(input.size() > k_fitting, "Input too small");
    input.resize(input.size() + k_fitting);
    kamping::Span<InputType> shift_span(std::prev(input.end(), k_fitting), k_fitting);
    kamping::Span<InputType> recv_span(std::prev(input.end(), k_fitting), k_fitting);

    comm.isend(
        kamping::send_buf(shift_span),
        kamping::destination(comm.rank_shifted_cyclic(-1)),
        kamping::send_count(k_fitting)
    );

    comm.recv(
        kamping::recv_buf(recv_span),
        kamping::source(comm.rank_shifted_cyclic(1)),
        kamping::recv_count(k_fitting)
    );

    size_t const local_size = input.size();
    auto         offset     = comm.exscan_single(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>()));

    std::vector<IRR<IndexType>> result;
    result.reserve(input.size());
    for (size_t i = 0; i < local_size; ++i) {
        IndexType rank1{hist[input[i]]};
        IndexType rank2{hist[input[i + k_fitting]]};
        for (size_t j = 0; j < k_fitting; ++j) {
            rank1 = (rank1 << bits_per_symbol) | hist[input[i + j]];
            rank2 = (rank2 << bits_per_symbol) | hist[input[i + k_fitting + j]];
        }
        result.emplace_back(offset++, rank1, rank2);
    }

    size_t const starting_iteration = std::floor(std::log2(k_fitting)) + 1;
    return std::pair{starting_iteration, result};
}

template <typename IndexType>
std::vector<IndexType> prefix_doubling(std::vector<uint8_t>&& input, kamping::BasicCommunicator& comm) {
    using seed_type = std::mt19937::result_type;
    auto local_seed =
        42 + static_cast<seed_type>(kamping::world_rank()) + static_cast<seed_type>(kamping::world_size());

    auto [irrs, iteration] = reduce_alphabet<IndexType>(std::move(input));

    while (true) {
    }
}
