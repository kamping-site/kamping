// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include "kamping/data_buffer.hpp"
#include "kamping/parameter_objects.hpp"
#include "legacy_parameter_objects.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    using ContainerType                     = std::vector<int>;
    ParameterType const      parameter_type = ParameterType::recv_buf;
    BufferType const         buffer_type    = BufferType::out_buffer;
    BufferResizePolicy const resize_policy  = BufferResizePolicy::resize_to_fit;

    ContainerType const                                                   const_container;
    ContainerBasedConstBuffer<ContainerType, parameter_type, buffer_type> container_based_const_buffer(const_container);

    ContainerBasedOwningBuffer<ContainerType, parameter_type, buffer_type> container_based_owning_buffer(
        std::vector<int>{1, 2, 3}
    );

    SingleElementConstBuffer<int, parameter_type, buffer_type> single_elem_const_buffer(42);

    SingleElementOwningBuffer<int, parameter_type, buffer_type> single_elem_owning_buffer(42);

    int                                                             elem = 42;
    SingleElementModifiableBuffer<int, parameter_type, buffer_type> single_elem_modifiable_buffer(elem);

    LibAllocatedSingleElementBuffer<int, parameter_type, buffer_type> lib_alloc_single_element_buffer;

    ContainerType container;
    UserAllocatedContainerBasedBuffer<ContainerType, parameter_type, buffer_type, resize_policy>
        user_alloc_container_based_buffer(container);

    LibAllocatedContainerBasedBuffer<ContainerType, parameter_type, buffer_type> lib_alloc_container_based_buffer;

    RootDataBuffer root(42);

#if defined(COPY_CONSTRUCT_CONTAINER_CONST_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = container_based_owning_buffer;
#elif defined(COPY_ASSIGN_CONTAINER_CONST_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    container_based_const_buffer = container_based_const_buffer;
#elif defined(COPY_CONSTRUCT_SINGLE_ELMENT_CONST_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = single_elem_owning_buffer;
#elif defined(COPY_ASSIGN_SINGLE_ELMENT_CONST_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    single_elem_const_buffer = single_elem_const_buffer;
#elif defined(COPY_CONSTRUCT_SINGLE_ELMENT_MODIFIABLE_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = lib_alloc_single_element_buffer;
#elif defined(COPY_ASSIGN_SINGLE_ELMENT_MODIFIABLE_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    single_elem_modifiable_buffer = single_elem_modifiable_buffer;
#elif defined(COPY_ASSIGN_USER_ALLOC_CONTAINER_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    user_alloc_container_based_buffer = user_alloc_container_based_buffer;
#elif defined(COPY_CONSTRUCT_LIB_ALLOC_CONTAINER_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = lib_alloc_container_based_buffer;
#elif defined(COPY_ASSIGN_LIB_ALLOC_CONTAINER_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    lib_alloc_container_based_buffer = lib_alloc_container_based_buffer;
#elif defined(COPY_CONSTRUCT_LIB_ALLOC_SINGLE_ELEMENT_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = lib_alloc_single_element_buffer;
#elif defined(COPY_ASSIGN_LIB_ALLOC_SINGLE_ELEMENT_BUFFER)
    // should not be possible to copy assign a buffer (for performance reasons)
    lib_alloc_single_element_buffer = lib_alloc_single_element_buffer;
#elif defined(COPY_CONSTRUCT_ROOT_BUFFER)
    // should not be possible to copy construct an owning buffer (for performance reasons)
    auto tmp = root;
#elif defined(COPY_ASSIGN_ROOT_BUFFER)
    // should not be possible to copy assign an owning buffer (for performance reasons)
    root = root;
#elif defined(VALUE_CONSTRUCTOR_REFERENCING_DATA_BUFFER)
    // should not be possible to value (or rvalue) construct a referencing DataBuffer
    DataBuffer<std::vector<int>, ParameterType::send_buf, BufferModifiability::modifiable, BufferOwnership::referencing>
        foo{std::vector<int>()};
#elif defined(DEFAULT_CONSTRUCT_USER_ALLOCATED_DATA_BUFFER)
    // should not be possible to default construct a user defined DataBuffer
    DataBuffer<
        std::vector<int>,
        ParameterType::send_buf,
        BufferModifiability::modifiable,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize>
        foo{};
#elif defined(EXTRACT_USER_ALLOCATED_DATA_BUFFER)
    // should not be possible to extract a user allocated DataBuffer
    DataBuffer<
        std::vector<int>,
        ParameterType::send_buf,
        BufferModifiability::modifiable,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize>
         foo{std::vector<int>()};
    auto bar = foo.extract();
#elif defined(RESIZE_CONST_DATA_BUFFER)
    // should not be possible to resize a constant DataBuffer
    DataBuffer<
        std::vector<int>,
        ParameterType::send_buf,
        BufferModifiability::constant,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize>
         foo{std::vector<int>()};
    auto bar = foo.resize(0);
#elif defined(GET_SINGLE_ELEMENT_ON_VECTOR)
    // should not be possible to call `get_single_element()` on a container based buffer
    DataBuffer<
        std::vector<int>,
        ParameterType::send_buf,
        BufferModifiability::constant,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize>
        foo{std::vector<int>()};
    foo.get_single_element();
#elif defined(ACCESS_CONST_VECTOR_BOOL)
    // should not be possible to do something useful with a container based on std::vector<bool>
    DataBuffer<
        std::vector<bool>,
        ParameterType::send_buf,
        BufferModifiability::constant,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize> const foo{std::vector<bool>()};
    foo.underlying();
#elif defined(ACCESS_VECTOR_BOOL)
    // should not be possible to do something useful with a container based on std::vector<bool>
    DataBuffer<
        std::vector<bool>,
        ParameterType::send_buf,
        BufferModifiability::modifiable,
        BufferOwnership::owning,
        BufferAllocation::user_allocated,
        BufferResizePolicy::no_resize>
        foo{std::vector<bool>()};
    foo.underlying();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
