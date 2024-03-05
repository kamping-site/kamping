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

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/scan.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

namespace kamping::plugin {

template <typename Comm, template <typename...> typename DefaultContainerType>
class SampleSort : public plugin::PluginBase<Comm, DefaultContainerType, SampleSort> {
public:
    template <typename RandomIt, typename Compare = std::less<typename std::iterator_traits<RandomIt>::value_type>>
    void sort(RandomIt begin, RandomIt end, Compare comp = Compare{}) {
        using ValueType = typename std::iterator_traits<RandomIt>::value_type;
        auto& self      = this->to_communicator();

        auto number_elements = std::distance(begin, end);

        size_t const           oversampling_ratio = 16 * static_cast<size_t>(std::log2(self.size())) + 1;
        std::vector<ValueType> local_samples(oversampling_ratio);
        std::sample(begin, end, local_samples.begin(), oversampling_ratio, std::mt19937{self.rank() + self.size()});

        auto global_samples = self.allgather(send_buf(local_samples));
        pick_splitters(self.size() - 1, oversampling_ratio, global_samples, comp);
        auto                   buckets = build_buckets(begin, end, global_samples, comp);
        std::vector<int>       scounts;
        std::vector<ValueType> data;
        data.reserve(number_elements);

        for (auto& bucket: buckets) {
            data.insert(data.end(), bucket.begin(), bucket.end());
            scounts.push_back(static_cast<int>(bucket.size()));
        }
        data = self.alltoallv(send_buf(data), send_counts(scounts));
        std::sort(data.begin(), data.end(), comp);

        data = balance_data(data, number_elements);
        for (size_t i = 0; i < data.size(); ++i, ++begin) {
            *begin = data[i];
        }
    }

private:
    template <typename T, typename Compare>
    void pick_splitters(size_t num_splitters, size_t oversampling_ratio, std::vector<T>& global_samples, Compare comp) {
        std::sort(global_samples.begin(), global_samples.end(), comp);
        for (size_t i = 0; i < num_splitters; i++) {
            global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
        }

        global_samples.resize(num_splitters);
    }

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
            buckets[std::distance(splitters.begin(), bound)].push_back(*it);
        }
        return buckets;
    }

    template <typename T>
    auto balance_data(std::vector<T>& data, size_t const target_size) {
        auto& self = this->to_communicator();

        size_t local_size     = data.size();
        size_t preceding_size = self.scan_single(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>()));

        auto target_sizes = self.allgather(kamping::send_buf(target_size));
        std::vector<typename decltype(target_sizes)::value_type> target_preceding_sizes;
        std::inclusive_scan(target_sizes.begin(), target_sizes.end(), std::back_inserter(target_preceding_sizes));

        auto get_target_rank = [&](const size_t pos) {
            size_t rank = 0;
            while (rank < self.size() && target_preceding_sizes[rank] < pos) {
                ++rank;
            }
            return rank;
        };

        std::vector<int32_t> send_counts(self.size(), 0);
        for (auto cur_rank = get_target_rank(preceding_size); local_size > 0 && cur_rank < self.size(); ++cur_rank) {
            const size_t to_send  = std::min(target_sizes[cur_rank], local_size);
            send_counts[cur_rank] = to_send;
            local_size -= to_send;
            preceding_size += to_send;
        }
        send_counts.back() += local_size;

        auto result = self.alltoallv(kamping::send_buf(data), kamping::send_counts(send_counts));
        return result;
    }
};

} // namespace kamping::plugin
