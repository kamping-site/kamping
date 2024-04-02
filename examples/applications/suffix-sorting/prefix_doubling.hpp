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

/// @brief Reducing the size of the alphabet and packing multiple characters together without decreasing the size of the
/// input.
///
/// Since the prefix doubling algorithm considers prefixes of all suffixes, starting with a long prefix is benefitial.
/// To this end, this function computes the reduced alphabet, i.e., maps every character to [1..\sigma], where \sigma is
/// the number of unique characters in the text. Then, the number of bits \ceil{\log\sigma} necessary to store a
/// character is determined. Afterwards, the longest prefix of each suffix, i.e., a prefix of length
/// |IndexType|/\ceil{\log\sigma}, is stored packed into one integer of type \c IndexType.
///
/// @tparam IndexType The data type used to represent entries in the suffix array.
/// @param input The input text.
/// @param comm The KaMPIng communicator used for this algorithms.
/// @return A tuple consisting of the starting iteration a prefix doubling algorithm can start with due to the packed
/// alphabet and the tuples that contain all information for the first iteration of the prefix doubling algorithm.
template <typename IndexType>
auto reduce_alphabet(
    std::vector<std::uint8_t>&& input, kamping::Communicator<std::vector, kamping::plugin::SampleSort>& comm
) {
    using namespace kamping;

    using InputType = std::remove_reference_t<decltype(input)>::value_type;

    std::array<IndexType, std::numeric_limits<InputType>::max() + 1> hist = {0};

    // Compute histograms of characters in local input.
    std::for_each(input.begin(), input.end(), [&hist](auto const symbol) { ++hist[symbol]; });

    // Sum up all histograms on all PEs.
    comm.allreduce_inplace(send_recv_buf(hist), op(ops::plus<>()));

    size_t new_alphabet_size = 1;
    std::transform(hist.cbegin(), hist.cend(), hist.begin(), [&new_alphabet_size](auto const count) {
        return count > 0 ? new_alphabet_size++ : 0;
    });

    // Determine the bits necessary to store a character in the reduced alphabet.
    size_t const bits_per_symbol = static_cast<size_t>(std::ceil(std::log2(new_alphabet_size)));

    // Number of characters fitting in one IndexType.
    size_t const k_fitting = (CHAR_BIT * sizeof(IndexType)) / bits_per_symbol;

    // Prepare sending 2*k_fitting characters to the preceding PE.
    KASSERT(input.size() > 2 * k_fitting, "Input too small");
    size_t const local_size = input.size();
    input.resize(input.size() + (2 * k_fitting), 0);
    Span<InputType> shift_span(
        std::next(input.begin(), static_cast<int64_t>(local_size - (2 * k_fitting))),
        2 * k_fitting
    );
    Span<InputType> recv_span(std::next(input.begin(), static_cast<int64_t>(local_size)), 2 * k_fitting);

    if (comm.rank() != 0) {
        comm.isend(send_buf(shift_span), destination(comm.rank_shifted_cyclic(-1)));
    }

    // Receive the 2*k_fitting characters from the succeeding PE.
    if (comm.rank() + 1 < comm.size()) {
        comm.recv(recv_buf(recv_span), source(comm.rank_shifted_cyclic(1)));
    }

    // Pack the alphabet.
    auto                        offset = comm.exscan_single(send_buf(local_size), op(ops::plus<>()));
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

    // Return the iteration a prefix doubling algorithm can start in and the input with reduced alphabet.
    size_t const starting_iteration = static_cast<size_t>(std::floor(std::log2(k_fitting))) + 1;
    return std::pair{starting_iteration, result};
}

/// @brief Compute the suffix array by sorting length-2^k prefixes of suffixes lexicographically until all prefixes are
/// unique. This requires O(\log n) iterations.
///
/// When sorting the length-2^k prefixes, instead of comparing them character by character, the new order is determined
/// by the order of the length-2^{k-1} prefixes, which have been computed in the previous iteration. The suffix array
/// can therefore be computed without comparing any suffix character by character. To obtain the new order of prefixes,
/// the order of the prefixes 2^{k-1} text positions apart are needed. Since these prefixes can be on different PEs,
/// this algorithm utilizes sorting to put the prefixes (or at least their rank, i.e., their order) next to each other.
/// This technique has first been presented in external memory by Dementiev et al. in "Better External Memory Suffix
/// Array Construction", ACM J. Exp. Algorithmics, 2008.
///
/// @tparam IndexType Index type of the suffix array.
/// @param input Local input for which the suffix array is computed.
/// @param comm KaMPIng communicator used for this algorithm.
/// @return The suffix array for the concatenation of all \c inputs.
template <typename IndexType>
std::vector<IndexType>
prefix_doubling(std::vector<uint8_t>&& input, kamping::Communicator<std::vector, kamping::plugin::SampleSort>& comm) {
    using namespace kamping;
    // Reduce the alphabet to start with prefixes longer than 1.
    auto [start_iteration, irrs] = reduce_alphabet<IndexType>(std::move(input), comm);
    IndexType iteration =
        asserting_cast<IndexType>(start_iteration); // clang does not allow to capture structured bindings in lambdas

    ///
    size_t                     local_size = irrs.size();
    size_t                     offset     = 0;
    std::vector<IR<IndexType>> irs;

    // This loop is broken when all prefixes are unique, i.e., the suffix array has been computed.
    while (true) {
        // Sort rank tuples to determine new ranks for each position.
        comm.sort(irrs, [](IRR<IndexType> const& lhs, IRR<IndexType> const& rhs) {
            return std::tie(lhs.rank1, lhs.rank2) < std::tie(rhs.rank1, rhs.rank2);
        });
        local_size = irrs.size();
        offset     = comm.exscan_single(send_buf(local_size), op(ops::plus<>()), values_on_rank_0({0}));

        irs.clear();
        irs.reserve(local_size);

        // Compute new ranks.
        size_t cur_rank = offset;
        irs.emplace_back(irrs[0].index, cur_rank);
        for (size_t i = 1; i < local_size; ++i) {
            if (irrs[i - 1] != irrs[i]) {
                cur_rank = offset + i;
            }
            irs.emplace_back(irrs[i].index, cur_rank);
        }

        // Determine if all ranks are unique *locally*.
        bool all_distinct = true;
        for (size_t i = 1; i < local_size; ++i) {
            all_distinct &= (irs[i].rank != irs[i - 1].rank);
            if (!all_distinct) {
                break;
            }
        }

        // Determine if all ranks are unique *globally*.
        all_distinct = comm.allreduce_single(send_buf(all_distinct), op(ops::logical_and<>()));
        if (all_distinct) {
            break;
        }

        // Sort ranks such that ranks needed to compute new ranks in the next iteration are next to each other.
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
        offset     = comm.exscan_single(send_buf(local_size), op(ops::plus<>()));

        comm.isend(send_buf(irs.front()), destination(comm.rank_shifted_cyclic(-1)), send_count(1));

        IR<IndexType> rightmost_ir;
        comm.recv(recv_buf(rightmost_ir), source(comm.rank_shifted_cyclic(1)), recv_count(1));

        if (comm.rank() + 1 < comm.size()) {
            irs.push_back(rightmost_ir);
        } else {
            irs.emplace_back(0, 0);
        }

        irrs.clear();
        irrs.reserve(local_size);
        IndexType const index_distance = IndexType{1} << iteration;

        // Compute new rank pairs.
        for (size_t i = 0; i < local_size; ++i) {
            IndexType second_rank{0};
            if (irs[i].index + index_distance == irs[i + 1].index) {
                second_rank = irs[i + 1].rank;
            }
            irrs.emplace_back(irs[i].index, irs[i].rank, second_rank);
        }
        ++iteration;
    }
    // Transform the ranks of the suffixes (the inverse suffix array) to the suffix array.
    std::vector<IndexType> result;
    result.reserve(irs.size());
    std::transform(irs.begin(), irs.end(), std::back_inserter(result), [](IR<IndexType> const& ir) {
        return ir.index;
    });
    return result;
}
