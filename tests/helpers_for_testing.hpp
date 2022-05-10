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

/// @file
/// @brief Some mock objects etc. used in multiple test scenarios.

#pragma once

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace testing {
/// @brief Simple Container type. Can be used to test library function with containers other than vector.
///
template <typename T>
class OwnContainer {
public:
    using value_type = T;

    OwnContainer() = default;
    OwnContainer(size_t size) : _vec(size) {}
    OwnContainer(OwnContainer<T> const& rhs) : _vec(rhs._vec), _copy_count(rhs._copy_count) {
        (*_copy_count)++;
    }
    OwnContainer(OwnContainer<T>&& rhs) : _vec(std::move(rhs._vec)), _copy_count(rhs._copy_count) {}
    OwnContainer<T>& operator=(OwnContainer<T> const& rhs) {
        this->_vec        = rhs._vec;
        this->_copy_count = rhs._copy_count;
        (*_copy_count)++;
        return *this;
    }

    T* data() noexcept {
        return _vec.data();
    }

    const T* data() const noexcept {
        return _vec.data();
    }

    std::size_t size() const {
        return _vec.size();
    }

    void resize(std::size_t new_size) {
        _vec.resize(new_size);
    }

    const T& operator[](size_t i) const {
        return _vec[i];
    }

    T& operator[](size_t i) {
        return _vec[i];
    }

    size_t copy_count() const {
        return *_copy_count;
    }

    bool operator==(const OwnContainer<T>& other) const {
        return _vec == other._vec;
    }

private:
    std::vector<T>          _vec;
    std::shared_ptr<size_t> _copy_count = 0;
};

/// @ Mock argument for wrapped \c MPI calls.
template <kamping::internal::ParameterType _parameter_type>
struct Argument {
    static constexpr kamping::internal::ParameterType parameter_type = _parameter_type;
    Argument(int i) : _i{i} {}
    int _i;
};

/// @brief Custom expectation for testing if a KASSERT fails.
#define EXPECT_KASSERT_FAILS(CODE, FAILURE_MESSAGE) \
    EXPECT_EXIT({ CODE; }, testing::KilledBySignal(SIGABRT), FAILURE_MESSAGE);

/// @brief Custom assertion for testing if a KASSERT fails.
#define ASSERT_KASSERT_FAILS(CODE, FAILURE_MESSAGE) \
    ASSERT_EXIT({ CODE; }, testing::KilledBySignal(SIGABRT), FAILURE_MESSAGE);

/// @}
} // namespace testing
