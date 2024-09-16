// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "./helpers_for_testing.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;
using namespace ::testing;

TEST(ParameterObjectsTest, tag_param_clone) {
    TagParam<TagType::value> value_tag(5);
    TagParam<TagType::value> value_tag_clone = value_tag.clone();
    EXPECT_EQ(value_tag.parameter_type, value_tag_clone.parameter_type);
    EXPECT_EQ(value_tag.tag_type, value_tag_clone.tag_type);
    EXPECT_EQ(value_tag.tag(), value_tag_clone.tag());

    TagParam<TagType::any> any_tag;
    TagParam<TagType::any> any_tag_clone = any_tag.clone();
    EXPECT_EQ(any_tag.parameter_type, any_tag_clone.parameter_type);
    EXPECT_EQ(any_tag.tag_type, any_tag_clone.tag_type);
    EXPECT_EQ(any_tag.tag(), any_tag_clone.tag());
}

TEST(ParameterObjectsTest, rank_data_buffer_clone) {
    RankDataBuffer<RankType::value, ParameterType::source> value_rank(5);
    RankDataBuffer<RankType::value, ParameterType::source> value_rank_clone = value_rank.clone();
    EXPECT_EQ(value_rank.parameter_type, value_rank_clone.parameter_type);
    EXPECT_EQ(value_rank.rank_type, value_rank_clone.rank_type);
    EXPECT_EQ(value_rank.rank_signed(), value_rank_clone.rank_signed());

    RankDataBuffer<RankType::any, ParameterType::source> any_rank;
    RankDataBuffer<RankType::any, ParameterType::source> any_rank_clone = any_rank.clone();
    EXPECT_EQ(any_rank.parameter_type, any_rank_clone.parameter_type);
    EXPECT_EQ(any_rank.rank_type, any_rank_clone.rank_type);
    EXPECT_EQ(any_rank.rank_signed(), any_rank_clone.rank_signed());

    RankDataBuffer<RankType::null, ParameterType::source> null_rank;
    RankDataBuffer<RankType::null, ParameterType::source> null_rank_clone = null_rank.clone();
    EXPECT_EQ(null_rank.parameter_type, null_rank_clone.parameter_type);
    EXPECT_EQ(null_rank.rank_type, null_rank_clone.rank_type);
    EXPECT_EQ(null_rank.rank_signed(), null_rank_clone.rank_signed());
}

TEST(ParameterObjectsTest, DataBufferBuilder_with_noncopyable_type) {
    { // by reference
        testing::NonCopyableOwnContainer<int> container{1, 2, 3, 4};
        auto                                  b = make_data_buffer_builder<
            kamping::internal::ParameterType::recv_buf,
            kamping::internal::BufferModifiability::modifiable,
            kamping::internal::BufferType::out_buffer,
            resize_to_fit>(container);
        container[0] = 42;
        EXPECT_EQ(b.size(), 4);
        auto buffer = b.construct_buffer_or_rebind();
        EXPECT_THAT(buffer.underlying(), ElementsAre(42, 2, 3, 4));
    }

    { // by value
        testing::NonCopyableOwnContainer<int> container{1, 2, 3, 4};
        auto                                  b = make_data_buffer_builder<
            kamping::internal::ParameterType::recv_buf,
            kamping::internal::BufferModifiability::modifiable,
            kamping::internal::BufferType::out_buffer,
            resize_to_fit>(std::move(container));
        container.resize(1);
        container[0] = 42;
        EXPECT_EQ(b.size(), 4);
        auto buffer = b.construct_buffer_or_rebind();
        EXPECT_THAT(buffer.underlying(), ElementsAre(1, 2, 3, 4));
    }
    { // rebind
        auto b = make_data_buffer_builder<
            kamping::internal::ParameterType::recv_buf,
            kamping::internal::BufferModifiability::modifiable,
            kamping::internal::BufferType::out_buffer,
            resize_to_fit>(alloc_container_of<int>);
        auto buffer = b.template construct_buffer_or_rebind<NonCopyableOwnContainer>();
        EXPECT_TRUE(is_non_copyable_own_container<std::remove_reference_t<decltype(buffer.underlying())>>);
    }
}

TEST(ParameterObjectsTest, DataBufferBuilder_with_const_owning_noncopyable_type) {
    {
        testing::NonCopyableOwnContainer<int> container{1, 2, 3, 4};
        auto                                  b = make_data_buffer_builder<
            kamping::internal::ParameterType::recv_buf,
            kamping::internal::BufferModifiability::constant,
            kamping::internal::BufferType::out_buffer,
            no_resize>(std::move(container));
        container.resize(1);
        container[0] = 42;
        EXPECT_EQ(b.size(), 4);
        auto           buffer                 = b.construct_buffer_or_rebind();
        constexpr bool is_data_ptr_constant   = std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>;
        constexpr bool is_underlying_constant = std::is_const_v<std::remove_reference_t<decltype(buffer.underlying())>>;
        EXPECT_TRUE(is_data_ptr_constant);
        EXPECT_TRUE(is_underlying_constant);
        EXPECT_THAT(buffer.underlying(), ElementsAre(1, 2, 3, 4));
        EXPECT_THAT(buffer.extract(), ElementsAre(1, 2, 3, 4));
    }
}
