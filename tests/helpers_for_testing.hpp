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

#include <mpi.h>

#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

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

    T const* data() const noexcept {
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

    T const& operator[](size_t i) const {
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

    T const* cbegin() const {
        return _data;
    }

    T const* cend() const {
        return _data + _size;
    }

private:
    T*                      _data;
    size_t                  _size;
    std::shared_ptr<size_t> _copy_count;
};

/// @brief Simple non-copyable container type.
///
template <typename T>
class NonCopyableOwnContainer : public testing::OwnContainer<T> {
public:
    using testing::OwnContainer<T>::OwnContainer;

    NonCopyableOwnContainer(NonCopyableOwnContainer<T> const&) = delete;
    NonCopyableOwnContainer(NonCopyableOwnContainer<T>&&)      = default;

    NonCopyableOwnContainer<T>& operator=(NonCopyableOwnContainer<T> const&) = delete;
    NonCopyableOwnContainer<T>& operator=(NonCopyableOwnContainer<T>&&)      = default;
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

///@brief Returns an uncommitted MPI_Datatype with type signature {int, sizeof(int)-padding, int}.
///@return created MPI_Datatype.
inline MPI_Datatype MPI_INT_padding_MPI_INT() {
    MPI_Datatype new_type;
    // create 2 blocks of length 1 (MPI_INT) with a stride (distance between the start of each block in number elems)
    // of 2
    MPI_Type_vector(2, 1, 2, MPI_INT, &new_type);
    return new_type;
}

///@brief Returns an uncommitted MPI_Datatype with type signature {int, sizeof(int)-padding, sizeof(int)-padding}.
///@return created MPI_Datatype.
inline MPI_Datatype MPI_INT_padding_padding() {
    MPI_Datatype new_type;
    MPI_Type_create_resized(MPI_INT, 0, sizeof(int) * 3, &new_type);
    return new_type;
}

///@ This allows to start a dummy non-blocking operation, which can be manually marked as completed.
struct DummyNonBlockingOperation {
    template <typename... Args>
    ///@brief Starts a dummy non-blocking operation using MPI's generalized requests.
    /// The value of the tag is stored in the receive buffer and as tag in the status returned after completion.
    ///
    /// Optional parameters:
    ///- \c tag: The tag of the operation. Defaults to 0.
    ///- \c request: The request object to use. Defaults to internally constructed \c kamping::request() which is return
    /// to the user.
    ///- \c recv_buf: The receive buffer to use. Defaults to \c kamping::recv_buf(alloc_new<std::vector<int>>).
    ///@param args Arguments to the operation.
    auto start_op(Args... args) {
        using namespace kamping::internal;
        using namespace kamping;
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(),
            KAMPING_OPTIONAL_PARAMETERS(tag, request, recv_buf)
        );
        using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<std::vector<int>>));
        auto&& recv_buf =
            select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...)
                .get();
        using recv_buf_type       = typename std::remove_reference_t<decltype(recv_buf)>;
        using recv_buf_value_type = typename recv_buf_type::value_type;
        static_assert(std::is_same_v<recv_buf_value_type, int>);
        auto compute_required_recv_buf_size = [&] {
            return size_t{1};
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        using default_request_param = decltype(kamping::request());
        auto&& request_param =
            select_parameter_type_or_default<ParameterType::request, default_request_param>(std::tuple{}, args...);

        using default_tag_buf_type = decltype(kamping::tag(0));

        auto&& tag_param =
            select_parameter_type_or_default<ParameterType::tag, default_tag_buf_type>(std::tuple(0), args...);
        int tag     = tag_param.tag();
        this->state = new int(tag);
        this->data  = recv_buf.get().data();
        MPI_Grequest_start(
            [](void* extra_state [[maybe_unused]], MPI_Status* status [[maybe_unused]]) {
                MPI_Status_set_elements(status, MPI_INT, 1);
                MPI_Status_set_cancelled(status, 0);
                MPI_Comm_rank(MPI_COMM_WORLD, &status->MPI_SOURCE);
                status->MPI_TAG = *static_cast<int*>(extra_state);
                return MPI_SUCCESS;
            },
            [](void* extra_state [[maybe_unused]]) {
                delete static_cast<int*>(extra_state);
                return MPI_SUCCESS;
            },
            [](void* extra_state [[maybe_unused]], int complete [[maybe_unused]]) { return MPI_SUCCESS; },
            this->state,
            &request_param.underlying().mpi_request()
        );
        this->req = request_param.underlying().mpi_request();
        return make_nonblocking_result(std::move(recv_buf), std::move(request_param));
    }

    ///@brief Manually marks the operation as completed.
    void finish_op() {
        *this->data = *state;
        MPI_Grequest_complete(this->req);
    }

    int*        state;
    int*        data;
    MPI_Request req;
};

/// @}
} // namespace testing
