// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <kamping/status.hpp>
#include <mpi.h>

namespace kamping {

struct StatusRefIterator {
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = StatusConstRef;
    using difference_type   = std::ptrdiff_t;
    using pointer           = void;
    using reference         = value_type;

    StatusRefIterator(MPI_Status const* status) : _status(status) {}

    [[nodiscard]] reference operator*() const {
        return StatusConstRef(*_status);
    }

    StatusRefIterator& operator++() {
        this->_status++;
        return *this;
    }

    StatusRefIterator& operator--() {
        this->_status--;
        return *this;
    }

    StatusRefIterator operator++(int) {
        auto tmp = *this;
        this->_status++;
        return tmp;
    }
    StatusRefIterator operator--(int) {
        auto tmp = *this;
        this->_status--;
        return tmp;
    }

    StatusRefIterator& operator+=(difference_type n) {
        this->_status += n;
        return *this;
    }

    StatusRefIterator& operator-=(difference_type n) {
        this->_status -= n;
        return *this;
    }

    [[nodiscard]] StatusRefIterator operator+(difference_type n) const {
        return StatusRefIterator(this->_status + n);
    }

    [[nodiscard]] StatusRefIterator operator-(difference_type n) const {
        return StatusRefIterator(this->_status - n);
    }

    [[nodiscard]] difference_type operator-(StatusRefIterator const& other) const {
        return this->_status - other._status;
    }

    [[nodiscard]] bool operator==(StatusRefIterator const& other) const {
        return _status == other._status;
    }

    [[nodiscard]] bool operator!=(StatusRefIterator const& other) const {
        return _status != other._status;
    }

    [[nodiscard]] bool operator<(StatusRefIterator const& other) const {
        return _status < other._status;
    }

    [[nodiscard]] bool operator>(StatusRefIterator const& other) const {
        return _status > other._status;
    }

    [[nodiscard]] bool operator<=(StatusRefIterator const& other) const {
        return _status <= other._status;
    }

    [[nodiscard]] bool operator>=(StatusRefIterator const& other) const {
        return _status >= other._status;
    }

private:
    MPI_Status const* _status;
};

class status_container_adaptor {
public:
    using value_type      = StatusConstRef;
    using const_reference = value_type;
    using const_iterator  = StatusRefIterator;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename Container>
    status_container_adaptor(Container const& container) {
        static_assert(
            std::is_same_v<typename Container::value_type, MPI_Status>,
            "Container must contain MPI_Status objects."
        );
        static_assert(internal::has_data_member_v<Container>, "Container must have a data member.");
        _data = container.data();
        _size = container.size();
    }
    [[nodiscard]] const_iterator begin() const noexcept {
        return StatusRefIterator(_data);
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return StatusRefIterator(_data + _size);
    }

    [[nodiscard]] const_reference operator[](size_type idx) const noexcept {
        return _data[idx];
    }

    [[nodiscard]] size_type size() const noexcept {
        return _size;
    }

    [[nodiscard]] bool empty() const noexcept {
        return _size == 0;
    }

private:
    MPI_Status const* _data;
    std::size_t       _size;
};
} // namespace kamping
