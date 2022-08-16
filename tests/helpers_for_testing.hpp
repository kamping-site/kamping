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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>

#include "kamping/assertion_levels.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace testing {
/// @brief Simple Container type. Can be used to test library function with containers other than vector.
///
template <typename T>
class OwnContainer {
public:
    using value_type     = T;
    using iterator       = T*;
    using const_iterator = T const*;

    OwnContainer() : OwnContainer(0) {}

    OwnContainer(size_t size) : OwnContainer(size, T{}) {}
    OwnContainer(size_t size, T value) : _data(nullptr), _size(size), _copy_count(std::make_shared<size_t>(0)) {
        _data = new T[_size];
        std::for_each(this->begin(), this->end(), [&value](T& val) { val = value; });
    }
    OwnContainer(std::initializer_list<T> elems)
        : _data(nullptr),
          _size(elems.size()),
          _copy_count(std::make_shared<size_t>(0)) {
        _data = new T[_size];
        std::copy(elems.begin(), elems.end(), _data);
    }
    OwnContainer(OwnContainer<T> const& rhs) : _data(nullptr), _size(rhs.size()), _copy_count(rhs._copy_count) {
        _data = new T[_size];
        std::copy(rhs.begin(), rhs.end(), _data);
        (*_copy_count)++;
    }
    OwnContainer(OwnContainer<T>&& rhs) : _data(rhs._data), _size(rhs._size), _copy_count(rhs._copy_count) {
        rhs._data       = nullptr;
        rhs._size       = 0;
        rhs._copy_count = std::make_shared<size_t>(0);
    }

    ~OwnContainer() {
        if (_data != nullptr) {
            delete[] _data;
            _data = nullptr;
        }
    }

    OwnContainer<T>& operator=(OwnContainer<T> const& rhs) {
        this->_data = new T[rhs._size];
        this->_size = rhs._size;
        std::copy(rhs.begin(), rhs.end(), _data);
        this->_copy_count = rhs._copy_count;
        (*_copy_count)++;
        return *this;
    }

    OwnContainer<T>& operator=(OwnContainer<T>&& rhs) {
        delete[] _data;
        _data           = rhs._data;
        _size           = rhs._size;
        _copy_count     = rhs._copy_count;
        rhs._data       = nullptr;
        rhs._size       = 0;
        rhs._copy_count = std::make_shared<size_t>(0);
        return *this;
    }

    T* data() noexcept {
        return _data;
    }

    const T* data() const noexcept {
        return _data;
    }

    std::size_t size() const {
        return _size;
    }

    void resize(std::size_t new_size) {
        if (new_size <= this->size()) {
            _size = new_size;
            return;
        }
        T* new_data = new T[new_size];
        std::copy(this->begin(), this->end(), new_data);
        std::for_each(new_data + this->size(), new_data + new_size, [](T& val) { val = T{}; });
        _size = new_size;
        delete[] _data;
        _data = new_data;
    }

    const T& operator[](size_t i) const {
        return _data[i];
    }

    T& operator[](size_t i) {
        return _data[i];
    }

    size_t copy_count() const {
        return *_copy_count;
    }

    bool operator==(OwnContainer<T> const& other) const {
        if (other.size() != this->size()) {
            return false;
        }
        for (size_t i = 0; i < _size; i++) {
            if (!(other[i] == this->operator[](i))) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(OwnContainer<T> const& other) const {
        return !(*this == other);
    }

    T* begin() const {
        return _data;
    }

    T* end() const {
        return _data + _size;
    }

    const T* cbegin() const {
        return _data;
    }

    const T* cend() const {
        return _data + _size;
    }

private:
    T*                      _data;
    size_t                  _size;
    std::shared_ptr<size_t> _copy_count;
};

/// @ Mock argument for wrapped \c MPI calls.
template <kamping::internal::ParameterType _parameter_type>
struct Argument {
    static constexpr kamping::internal::ParameterType parameter_type = _parameter_type;
    Argument(int i) : _i{i} {}
    int _i;
};

template <typename T>
struct CustomAllocator {
    using value_type = T;
    using pointer    = T*;
    using size_type  = size_t;

    CustomAllocator() = default;

    template <class U>
    constexpr CustomAllocator(CustomAllocator<U> const&) noexcept {}

    template <typename T1>
    struct rebind {
        using other = CustomAllocator<T1>;
    };

    pointer allocate(size_type n = 0) {
        return (pointer)malloc(n * sizeof(value_type));
    }
    void deallocate(pointer p, size_type) {
        free(p);
    }
};

//
// Makros to test for failed KASSERT() statements.
// Note that these makros could already be defined if we included the header that turns assertions into exceptions.
// In this case, we keep the current definition.
//

#ifndef EXPECT_KASSERT_FAILS
    #if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_HEAVY)
        // EXPECT that a KASSERT assertion failed and that the error message contains a certain failure_message.
        #define EXPECT_KASSERT_FAILS(code, failure_message) \
            EXPECT_EXIT({ code; }, testing::KilledBySignal(SIGABRT), failure_message);
    #else // Otherwise, we do not test for failed assertions
        #define EXPECT_KASSERT_FAILS(code, failure_message)
    #endif
#endif

#ifndef ASSERT_KASSERT_FAILS
    #if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_HEAVY)
        // ASSERT that a KASSERT assertion failed and that the error message contains a certain failure_message.
        #define ASSERT_KASSERT_FAILS(code, failure_message) \
            ASSERT_EXIT({ code; }, testing::KilledBySignal(SIGABRT), failure_message);
    #else // Otherwise, we do not test for failed assertions
        #define ASSERT_KASSERT_FAILS(code, failure_message)
    #endif
#endif

/// @}
} // namespace testing
