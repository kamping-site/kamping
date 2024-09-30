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
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "kamping/data_buffer.hpp"
#include "kamping/distributed_graph_communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

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

template <typename T>
static constexpr bool is_non_copyable_own_container = false;

template <typename T>
static constexpr bool is_non_copyable_own_container<NonCopyableOwnContainer<T>> = true;

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
                .construct_buffer_or_rebind();
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
        this->req         = request_param.underlying().mpi_request();
        using RecvBufType = std::remove_reference_t<decltype(recv_buf)>;
        return kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
            std::move(request_param),
            move_buffer_to_heap(std::move(recv_buf))
        );
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

///@brief Construct a container of size \param n, filled with values in [0, n)
///
///@tparam Container Type of container to be filled and returned.
///@param n Size of container.
///@param value Starting value to be filled into the container.
///@returns Container of size \param n filled with sequenitally increasing values starting with value.
template <typename Container = std::vector<int>>
auto iota_container_n(size_t n, typename Container::value_type value) {
    Container cont(n);
    std::iota(cont.begin(), cont.end(), value);
    return cont;
}

// Returns a std::vector containing all MPI_Datatypes equivalent to the given C++ datatype on this machine.
// Removes the topmost level of const qualifiers.
template <typename T>
std::vector<MPI_Datatype> possible_mpi_datatypes() noexcept {
    // Remove const qualifiers.
    using T_no_const = std::remove_const_t<T>;

    // Check if we got a array type -> create a continuous type.
    if constexpr (std::is_array_v<T_no_const>) {
        // sizeof(arrayType) returns the total length of the array not just the length of the first element. :-)
        // return std::vector<MPI_Datatype>{mpi_custom_continuous_type<sizeof(T_no_cv)>()};
        return std::vector<MPI_Datatype>{};
    }

    // Check if we got a enum type -> use underlying type
    if constexpr (std::is_enum_v<T_no_const>) {
        return possible_mpi_datatypes<std::underlying_type_t<T_no_const>>();
    }

    // For each supported C++ datatype, check if it is equivalent to the T_no_const and if so, add the corresponding MPI
    // datatype to the list of possible types.
    std::vector<MPI_Datatype> possible_mpi_datatypes;
    if constexpr (std::is_same_v<T_no_const, char>) {
        possible_mpi_datatypes.push_back(MPI_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, signed char>) {
        possible_mpi_datatypes.push_back(MPI_SIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned char>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_CHAR);
    }
    if constexpr (std::is_same_v<T_no_const, wchar_t>) {
        possible_mpi_datatypes.push_back(MPI_WCHAR);
    }
    if constexpr (std::is_same_v<T_no_const, signed short>) {
        possible_mpi_datatypes.push_back(MPI_SHORT);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned short>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_SHORT);
    }
    if constexpr (std::is_same_v<T_no_const, signed int>) {
        possible_mpi_datatypes.push_back(MPI_INT);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED);
    }
    if constexpr (std::is_same_v<T_no_const, signed long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, signed long long int>) {
        possible_mpi_datatypes.push_back(MPI_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, unsigned long long int>) {
        possible_mpi_datatypes.push_back(MPI_UNSIGNED_LONG_LONG);
    }
    if constexpr (std::is_same_v<T_no_const, float>) {
        possible_mpi_datatypes.push_back(MPI_FLOAT);
    }
    if constexpr (std::is_same_v<T_no_const, double>) {
        possible_mpi_datatypes.push_back(MPI_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_const, long double>) {
        possible_mpi_datatypes.push_back(MPI_LONG_DOUBLE);
    }
    if constexpr (std::is_same_v<T_no_const, int8_t>) {
        possible_mpi_datatypes.push_back(MPI_INT8_T);
    }
    if constexpr (std::is_same_v<T_no_const, int16_t>) {
        possible_mpi_datatypes.push_back(MPI_INT16_T);
    }
    if constexpr (std::is_same_v<T_no_const, int32_t>) {
        possible_mpi_datatypes.push_back(MPI_INT32_T);
    }
    if constexpr (std::is_same_v<T_no_const, int64_t>) {
        possible_mpi_datatypes.push_back(MPI_INT64_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint8_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT8_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint16_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT16_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint32_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT32_T);
    }
    if constexpr (std::is_same_v<T_no_const, uint64_t>) {
        possible_mpi_datatypes.push_back(MPI_UINT64_T);
    }
    if constexpr (std::is_same_v<T_no_const, bool>) {
        possible_mpi_datatypes.push_back(MPI_CXX_BOOL);
    }
    if constexpr (std::is_same_v<T_no_const, kamping::kabool>) {
        possible_mpi_datatypes.push_back(MPI_CXX_BOOL);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<float>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_FLOAT_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<double>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_DOUBLE_COMPLEX);
    }
    if constexpr (std::is_same_v<T_no_const, std::complex<long double>>) {
        possible_mpi_datatypes.push_back(MPI_CXX_LONG_DOUBLE_COMPLEX);
    }

    assert(possible_mpi_datatypes.size() > 0);
    return possible_mpi_datatypes;
}

/// @brief Compares two CommunicationGraphViews objects for equality
inline bool
are_equal(kamping::CommunicationGraphLocalView const& lhs, kamping::CommunicationGraphLocalView const& rhs) {
    auto compare_span = [](auto const& lhs_, auto const& rhs_) {
        if (lhs_.size() != rhs_.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs_.size(); ++i) {
            if (lhs_[i] != rhs_[i]) {
                return false;
            }
        }
        return true;
    };
    auto compare_optional_span = [&compare_span](auto const& lhs_, auto const& rhs_) {
        if (!lhs_.has_value() && !rhs_.has_value()) {
            return true;
        }
        if (lhs_.has_value() != rhs_.has_value()) {
            return false;
        }
        return compare_span(lhs_.value(), rhs_.value());
    };

    return compare_span(lhs.in_ranks(), rhs.in_ranks()) && compare_span(lhs.out_ranks(), rhs.out_ranks())
           && compare_optional_span(lhs.in_weights(), rhs.in_weights())
           && compare_optional_span(lhs.out_weights(), rhs.out_weights());
}

/// @}
} // namespace testing
