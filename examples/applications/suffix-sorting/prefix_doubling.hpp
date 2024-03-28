// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/plugin/sort.hpp"
#include "kassert/kassert.hpp"

template <typename IndexType>
struct IR {
    IndexType index;
    IndexType rank;

    IR() = default;

    IR(IndexType idx, IndexType r) : index(idx), rank(r) {}
}; // struct IR

template <typename IndexType>
struct IRR {
    IndexType index;
    IndexType rank1;
    IndexType rank2;

    IRR() = default;

    IRR(IndexType idx, IndexType r1, IndexType r2) : index(idx), rank1(r1), rank2(r2) {}

    bool operator<(IRR const& other) const {
        return std::tie(rank1, rank2) < std::tie(other.rank1, other.rank2);
    }

    bool operator!=(IRR const& other) const {
        return std::tie(rank1, rank2) != std::tie(other.rank1, other.rank2);
    }
}; // struct IRR

template <typename IndexType>
struct SATuple {
    IndexType rank;
    IndexType sa;
}; // struct SATuple

template <typename IndexType, typename SymbolType>
struct RRC {
    IndexType  rank1;
    IndexType  rank2;
    SymbolType symbol;
}; // struct RRC

template <typename IndexType>
auto reduce_alphabet(
    std::vector<std::uint8_t>&& input, kamping::Communicator<std::vector, kamping::plugin::SampleSort>& comm
) {
    using InputType = std::remove_reference_t<decltype(input)>::value_type;

    std::array<IndexType, std::numeric_limits<InputType>::max() + 1> hist = {0};

    std::for_each(input.begin(), input.end(), [&hist](auto const symbol) { ++hist[symbol]; });

    comm.allreduce_inplace(kamping::send_recv_buf(hist), kamping::op(kamping::ops::plus<>()));

    size_t new_alphabet_size = 1;
    std::transform(hist.cbegin(), hist.cend(), hist.begin(), [&new_alphabet_size](auto const count) {
        return count > 0 ? new_alphabet_size++ : 0;
    });

    size_t const bits_per_symbol = static_cast<size_t>(std::ceil(std::log2(new_alphabet_size)));

    size_t const k_fitting = (CHAR_BIT * sizeof(IndexType)) / bits_per_symbol;

    KASSERT(input.size() > 2 * k_fitting, "Input too small");
    size_t const local_size = input.size();
    input.resize(input.size() + (2 * k_fitting), 0);
    kamping::Span<InputType> shift_span(
        std::next(input.begin(), static_cast<int64_t>(local_size - (2 * k_fitting))),
        2 * k_fitting
    );
    kamping::Span<InputType> recv_span(std::next(input.begin(), static_cast<int64_t>(local_size)), 2 * k_fitting);

    if (comm.rank() != 0) {
        comm.isend(kamping::send_buf(shift_span), kamping::destination(comm.rank_shifted_cyclic(-1)));
    }

    if (comm.rank() + 1 < comm.size()) {
        comm.recv(kamping::recv_buf(recv_span), kamping::source(comm.rank_shifted_cyclic(1)));
    }

    auto offset = comm.exscan_single(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>()));
    std::vector<IRR<IndexType>> result;
    result.reserve(local_size);
    for (size_t i = 0; i < local_size; ++i) {
        IndexType rank1{hist[input[i]]};
        IndexType rank2{hist[input[i + k_fitting]]};
        for (size_t j = 0; j < k_fitting; ++j) {
            rank1 = (rank1 << bits_per_symbol) | hist[input[i + j]];
            rank2 = (rank2 << bits_per_symbol) | hist[input[i + k_fitting + j]];
        }
        result.emplace_back(offset++, rank1, rank2);
    }

    size_t const starting_iteration = static_cast<size_t>(std::floor(std::log2(k_fitting))) + 1;
    return std::pair{starting_iteration, result};
}

template <typename IndexType>
std::vector<IndexType>
prefix_doubling(std::vector<uint8_t>&& input, kamping::Communicator<std::vector, kamping::plugin::SampleSort>& comm) {
    auto [start_iteration, irrs] = reduce_alphabet<IndexType>(std::move(input), comm);
    IndexType iteration          = kamping::asserting_cast<IndexType>(start_iteration
    ); // clang does not allow to capture structured bindings in lambdas

    size_t                     local_size = irrs.size();
    size_t                     offset     = 0;
    std::vector<IR<IndexType>> irs;
    while (true) {
        comm.sort(irrs, [](IRR<IndexType> const& lhs, IRR<IndexType> const& rhs) {
            return std::tie(lhs.rank1, lhs.rank2) < std::tie(rhs.rank1, rhs.rank2);
        });
        local_size = irrs.size();
        offset     = comm.exscan_single(
            kamping::send_buf(local_size),
            kamping::op(kamping::ops::plus<>()),
            kamping::values_on_rank_0({0})
        );

        irs.clear();
        irs.reserve(local_size);

        size_t cur_rank = offset;
        irs.emplace_back(irrs[0].index, cur_rank);
        for (size_t i = 1; i < local_size; ++i) {
            if (irrs[i - 1] != irrs[i]) {
                cur_rank = offset + i;
            }
            irs.emplace_back(irrs[i].index, cur_rank);
        }

        bool all_distinct = true;
        for (size_t i = 1; i < local_size; ++i) {
            all_distinct &= (irs[i].rank != irs[i - 1].rank);
            if (!all_distinct) {
                break;
            }
        }

        all_distinct =
            comm.allreduce_single(kamping::send_buf(all_distinct), kamping::op(kamping::ops::logical_and<>()));
        if (all_distinct) {
            break;
        }

        comm.sort(irs, [=](IR<IndexType> const& lhs, IR<IndexType> const& rhs) {
            IndexType const mod_mask = (IndexType{1} << iteration) - 1;
            IndexType const div_mask = ~mod_mask;
            IndexType const lhs_mod  = lhs.index & mod_mask;
            IndexType const rhs_mod  = rhs.index & mod_mask;
            if (lhs_mod == rhs_mod) {
                return (lhs.index & div_mask) < (rhs.index & div_mask);
            } else {
                return lhs_mod < rhs_mod;
            }
        });

        local_size = irs.size();
        offset     = comm.exscan_single(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>()));

        comm.isend(
            kamping::send_buf(irs.front()),
            kamping::destination(comm.rank_shifted_cyclic(-1)),
            kamping::send_count(1)
        );

        IR<IndexType> rightmost_ir;
        comm.recv(
            kamping::recv_buf(rightmost_ir),
            kamping::source(comm.rank_shifted_cyclic(1)),
            kamping::recv_count(1)
        );

        if (comm.rank() + 1 < comm.size()) {
            irs.push_back(rightmost_ir);
        } else {
            irs.emplace_back(0, 0);
        }

        irrs.clear();
        irrs.reserve(local_size);
        IndexType const index_distance = IndexType{1} << iteration;

        for (size_t i = 0; i < local_size; ++i) {
            IndexType second_rank{0};
            if (irs[i].index + index_distance == irs[i + 1].index) {
                second_rank = irs[i + 1].rank;
            }
            irrs.emplace_back(irs[i].index, irs[i].rank, second_rank);
        }
        ++iteration;
    }
    std::vector<IndexType> result;
    result.reserve(irs.size());
    std::transform(irs.begin(), irs.end(), std::back_inserter(result), [](IR<IndexType> const& ir) {
        return ir.index;
    });
    return result;
}
