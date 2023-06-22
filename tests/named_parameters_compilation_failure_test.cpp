// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"

// a vector class using another value_type as the one given as first template parameter
template <typename T>
class FaultyVector : public std::vector<T> {
public:
    using value_type = float;
};

int main(int /*argc*/, char** /*argv*/) {
    using namespace kamping;
    constexpr internal::ParameterType type = internal::ParameterType::send_buf;

#ifdef VECTOR_BOOL_LVALUE
    // vector<bool> is not allowed
    std::vector<bool> v = {true, false};

    auto buf = internal::make_data_buffer<type, internal::BufferModifiability::modifiable>(v);
    buf.size();
#elif VECTOR_BOOL_RVALUE
    // vector<bool> is not allowed
    std::vector<bool> v = {true, false};

    auto buf = internal::make_data_buffer<type, internal::BufferModifiability::modifiable>(std::move(v));
    buf.size();
#elif VECTOR_BOOL_CUSTOM_ALLOCATOR
    // vector<bool> is not allowed, also if a custom allocator is used, because the STL may still do optimizations
    std::vector<bool, testing::CustomAllocator<bool> > v = {true, false};

    auto buf = internal::make_data_buffer<type, internal::BufferModifiability::modifiable>(v);
    buf.size();
#elif VECTOR_BOOL_NEW_CONTAINER
    // vector<bool> is not allowed
    std::vector<bool, testing::CustomAllocator<bool> > v = {true, false};

    auto buf =
        internal::make_data_buffer<type, internal::BufferModifiability::modifiable>(alloc_new<std::vector<bool> >{});
    buf.size();
#elif ALLOC_NEW_NOT_DEDUCTABLE
    // for recv_buf, the value type cannot be deduced
    auto buf = recv_buf(alloc_new_auto<std::vector>);
#elif FAULTY_VECTOR_ALLOC_NEW_AUTO
    // the faulty vector has the wrong value type
    auto buf = send_counts_out(alloc_new_auto<FaultyVector>);
// If none of the above sections is active, this file will compile successfully.
#endif
}
