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
#include <cstddef>
#include <random>
#include <vector>

using seed_type = std::mt19937::result_type;

template <typename T>
void pick_splitters(size_t num_splitters, size_t oversampling_ratio, std::vector<T>& global_samples) {
    std::sort(global_samples.begin(), global_samples.end());
    for (size_t i = 0; i < num_splitters; i++) {
        global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
    }
    global_samples.resize(num_splitters);
}

template <typename T>
auto build_buckets(std::vector<T>& data, std::vector<T>& splitters) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> buckets(splitters.size() + 1);
    for (auto& element: data) {
        auto const bound = std::upper_bound(splitters.begin(), splitters.end(), element);
        buckets[static_cast<size_t>(bound - splitters.begin())].push_back(element);
    }
    data.clear();
    return buckets;
}
