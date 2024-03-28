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
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include <kamping/utils/flatten.hpp>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/scan.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

namespace kamping::plugin {

/// @brief Plugin that adds a canonical sample sort to the communicator.
/// @tparam Type of the communicator that is extended by the plugin.
/// @tparam DefaultContainerType Default container type of the original communicator.
template <typename Comm, template <typename...> typename DefaultContainerType>
class SampleSort : public plugin::PluginBase<Comm, DefaultContainerType, SampleSort> {
public:
    /// @brief Sort the vector based on a binary comparison function (std::less by default).
    ///
    /// The order of equal elements is not guaranteed to be preserved. The binary comparison function has to be \c true
    /// if the first argument is less than the second.
    /// @tparam T Type of elements to be sorted.
    /// @tparam Allocator Allocator of the vector.
    /// @tparam Compare Type of the binary comparison function (\c std::less<T> by default).
    /// @param data Vector containing the data to be sorted.
    /// @param comp Binary comparison function used to determine the order of elements.
    template <typename T, typename Allocator, typename Compare = std::less<T>>
    void sort(std::vector<T, Allocator>& data, Compare comp = Compare{}) {
        auto&        self               = this->to_communicator();
        size_t const oversampling_ratio = 16 * static_cast<size_t>(std::log2(self.size())) + (data.size() > 0 ? 1 : 0);
        std::vector<T> local_samples(oversampling_ratio);
        std::sample(
            data.begin(),
            data.end(),
            local_samples.begin(),
            oversampling_ratio,
            std::mt19937{self.rank() + self.size()}
        );

        auto global_samples = self.allgatherv(send_buf(local_samples));
        pick_splitters(self.size() - 1, oversampling_ratio, global_samples, comp);
        auto buckets = build_buckets(data.begin(), data.end(), global_samples, comp);
        data = with_flattened(buckets).call([&](auto... flattened) { return self.alltoallv(std::move(flattened)...); });
        std::sort(data.begin(), data.end(), comp);
    }

    /// @brief Sort the elements in [begin, end) using a binary comparison function (std::less by default).
    ///
    /// The order of equal elements in not guaranteed to be preserved. The binary comparison function has to be \c true
    /// if the first argument is less than the second.
    /// @tparam RandomIt Iterator type of the container containing the elements that are sorted.
    /// @tparam OutputIt Iterator type of the output iterator.
    /// @tparam Compare Type of the binary comparison function (\c std::less<> by default).
    /// @param begin Start of the range of elements to sort.
    /// @param end Element after the last element to be sorted.
    /// @param out Output iterator used to output the sorted elements.
    /// @param comp Binary comparison function used to determine the order of elements.
    template <
        typename RandomIt,
        typename OutputIt,
        typename Compare = std::less<typename std::iterator_traits<RandomIt>::value_type>>
    void sort(RandomIt begin, RandomIt end, OutputIt out, Compare comp = Compare{}) {
        using ValueType = typename std::iterator_traits<RandomIt>::value_type;

        auto&        self               = this->to_communicator();
        size_t const local_size         = asserting_cast<size_t>(std::distance(begin, end));
        size_t const oversampling_ratio = 16 * static_cast<size_t>(std::log2(self.size())) + (local_size > 0 ? 1 : 0);
        std::vector<ValueType> local_samples(oversampling_ratio);
        std::sample(
            begin,
            end,
            local_samples.begin(),
            oversampling_ratio,
            std::mt19937{asserting_cast<std::mt19937::result_type>(self.rank() + self.size())}
        );

        auto global_samples = self.allgatherv(send_buf(local_samples));
        pick_splitters(self.size() - 1, oversampling_ratio, global_samples, comp);
        auto buckets = build_buckets(begin, end, global_samples, comp);
        auto data =
            with_flattened(buckets).call([&](auto... flattened) { return self.alltoallv(std::move(flattened)...); });
        std::sort(data.begin(), data.end(), comp);
        std::copy(data.begin(), data.end(), out);
    }

private:
    /// @brief Picks spliters from a global list of splitters.
    /// @tparam T Type of elements to be sorted (and of splitters)
    /// @tparam Compare Type of the binary comparison function used to determine order of elements.
    /// @param num_splitters Number of splitters that should be selected.
    /// @param oversampling_ratio Ratio at which local splitters are sampled.
    /// @param global_samples List of all (global) samples. Functions as out parameter where the picked samples are
    /// stored in.
    /// @param comp Binary comparison function used to determine order of elements.
    template <typename T, typename Compare>
    void pick_splitters(size_t num_splitters, size_t oversampling_ratio, std::vector<T>& global_samples, Compare comp) {
        std::sort(global_samples.begin(), global_samples.end(), comp);
        for (size_t i = 0; i < num_splitters; i++) {
            global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
        }
        global_samples.resize(num_splitters);
    }

    /// @brief Build buckets for a set of elements based on a set of splitters.
    /// @tparam RandomIt Iterator type used to iterate through the set of elements.
    /// @tparam T Type of elements.
    /// @tparam Compare Type of binary comparison function used to determine order of elements.
    /// @param begin Iterator to the beginning of the elements.
    /// @param end Iterator pointing behind the laste element.
    /// @param splitters
    template <typename RandomIt, typename T, typename Compare>
    auto build_buckets(RandomIt begin, RandomIt end, std::vector<T>& splitters, Compare comp)
        -> std::vector<std::vector<T>> {
        static_assert(
            std::is_same_v<T, typename std::iterator_traits<RandomIt>::value_type>,
            "Iterator value type and splitters do not match "
        );
        std::vector<std::vector<T>> buckets(splitters.size() + 1);
        for (auto it = begin; it != end; ++it) {
            auto const bound = std::upper_bound(splitters.begin(), splitters.end(), *it, comp);
            buckets[asserting_cast<size_t>(std::distance(splitters.begin(), bound))].push_back(*it);
        }
        return buckets;
    }
};

} // namespace kamping::plugin
