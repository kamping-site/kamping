// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <mdspan>

#include "kamping/data_buffer.hpp"

namespace kamping::adapter {

template <typename T, typename Extent, typename LayoutPolicy>
requires std::same_as<LayoutPolicy, std::layout_left> || std::same_as<LayoutPolicy, std::layout_right>
class MDSpanAdapter {
public:
    explicit MDSpanAdapter(std::mdspan<T, Extent, LayoutPolicy>& ms) : mdspan(ms), data_handle(mdspan.data_handle()) {}

    auto begin() noexcept {
        return data_handle;
    }
    auto end() noexcept {
        return data_handle + mdspan.size();
    }
    auto data() {
        return data_handle;
    }

    [[nodiscard]] size_t size() const {
        return mdspan.size();
    }

    auto get_mdspan() {
        return mdspan;
    }

private:
    std::mdspan<T, Extent> mdspan;
    T*                     data_handle;
};

} // namespace kamping::adapter
