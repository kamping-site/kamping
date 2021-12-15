/// @file
/// @brief Buffer wrapper around buffer based parameter types

#pragma once

#include <cstddef>
#include <memory>

#include "definitions.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{


/// @brief Type used for tag dispatching.
///
/// This types needs to be used to select internal::LibAllocContainerBasedBuffer as buffer type.
struct NewContainer {};
/// @brief Type used for tag dispatching.
///
/// This types needs to be used to select internal::LibAllocUniquePtrBasedBuffer as buffer type.
struct NewPtr {};

namespace internal {


///
/// @brief Own simple implementation of std::span.
///
/// Since KaMPI.ng needs to be C++17 compatible and std::span is part of C++20, we need our own implementation.
/// @tparam T type for which the span is defined.
template <typename T>
struct Span {
    const T* ptr;  ///< Pointer to the data reference by Span.
    size_t   size; ///< Number of elements of type T referenced by Span.
};

///
/// @brief Constant buffer based on on a pointer.
///
/// PtrBasedConstBuffer wraps read-only buffer storage of type T and represents an input of ParameterType type
/// @tparam T type contained in the buffer
/// @tparam ParameterType parameter type represented by this buffer
template <typename T, ParameterType type>
class PtrBasedConstBuffer {
public:
    static constexpr ParameterType ptype = type; ///< The type of parameter this buffer represents
    using value_type                     = T; ///< Value type of the buffer. //TODO this seems so obvious/redundant ...

    PtrBasedConstBuffer(const T* ptr, size_t size) : _span{ptr, size} {}

    ///@brief Get access to the underlying read-only storage.
    ///@return Span referring to the underlying read-only storage.
    Span<T> get() {
        return _span;
    }

private:
    Span<T> _span; ///< Actual storage to which PtrBasedConstBuffer refers.
};

template <typename Cont, ParameterType type>
class ContainerBasedConstBuffer {
public:
    static constexpr ParameterType ptype = type;
    using value_type                     = typename Cont::value_type;

    ContainerBasedConstBuffer(const Cont& cont) : _cont(cont) {}

    Span<value_type> get() {
        return {std::data(_cont), _cont.size()};
    }

private:
    const Cont& _cont;
};

///
/// @brief Struct containing sum definitions used by all modifiable buffers.
///
/// @tparam ParameterType (parameter) type represented by this buffer
/// @tparam is_consumable_ indicates whether this buffer already contains useable data
template <ParameterType type, bool is_consumable_>
struct BufferParameterType {
    static constexpr ParameterType ptype = type; ///< ParameterType which the buffer represents.
    static constexpr bool          is_consumable =
        is_consumable_; ///< This flag indicates whether the buffer content can be consumed or whether the underlying
                        ///< storage simply contains "empty" memory which needs to be filled.
};

template <typename Cont, ParameterType ptype, bool is_consumable>
class UserAllocContainerBasedBuffer : public BufferParameterType<ptype, is_consumable> {
public:
    using value_type = typename Cont::value_type;
    UserAllocContainerBasedBuffer(Cont& cont) : _cont(cont) {}
    value_type* get_ptr(size_t s) {
        if (_cont.size() < s)
            _cont.resize(s);
        return _cont.data();
    }

private:
    Cont& _cont;
};

template <typename T, ParameterType trait, bool is_consumable>
class UserAllocUniquePtrBasedBuffer : public BufferParameterType<trait, is_consumable> {
public:
    using value_type = T;
    UserAllocUniquePtrBasedBuffer(std::unique_ptr<T[]>& ptr) : ptr_ref(ptr) {}
    T* get_ptr([[maybe_unused]] size_t s) {
        return ptr_ref.get();
    }

private:
    std::unique_ptr<T[]>& ptr_ref;
};

template <typename Cont, ParameterType trait>
class LibAllocContainerBasedBuffer : public BufferParameterType<trait, false> {
public:
    using value_type = typename Cont::value_type;
    LibAllocContainerBasedBuffer() {}
    value_type* get_ptr(size_t s) {
        _cont.resize(s);
        return std::data(_cont);
    }
    Cont extract() {
        return std::move(_cont);
    }
    operator Cont() {
        return std::move(_cont);
    }

private:
    Cont _cont;
};

template <typename T, ParameterType ptype>
class LibAllocUniquePtrBasedBuffer : public BufferParameterType<ptype, false> {
public:
    using value_type = T;
    LibAllocUniquePtrBasedBuffer() {}
    T* get_ptr(size_t s) {
        _ptr_ref = std::make_unique<T[]>(s);
        return _ptr_ref.get();
    }
    std::unique_ptr<T[]> extract() {
        return std::move(_ptr_ref);
    }
    operator std::unique_ptr<T[]>() {
        return std::move(_ptr_ref);
    }

private:
    std::unique_ptr<T[]> _ptr_ref;
};

template <typename Cont, ParameterType trait, bool is_consumable>
class MovedContainerBasedBuffer : public BufferParameterType<trait, is_consumable> {
public:
    using value_type = typename Cont::value_type;
    MovedContainerBasedBuffer(Cont&& rref) : _cont(std::forward<Cont>(rref)) {}
    value_type* get_ptr(size_t s) {
        _cont.resize(s);
        return std::data(_cont);
    }
    Cont extract() {
        return std::move(_cont);
    }
    operator Cont() {
        return std::move(_cont);
    }

private:
    Cont _cont;
};

///@brief Macro to generate all buffers based on kamping::internal::ContainerBasedConstBuffer.
///
///@param func_name Name of the function that will be generated.
///@param trait kamping::internal::ParameterType which the buffer returned by the generated function will represent.
#define DEFINE_CONTAINER_BASED_CONST_BUFFER(func_name, trait)                                                     \
    template <typename Cont>                                                                                      \
    kamping::internal::ContainerBasedConstBuffer<Cont, kamping::internal::ParameterType::trait> func_name(        \
        Cont& cont) {                                                                                             \
        return kamping::internal::ContainerBasedConstBuffer<Cont, kamping::internal::ParameterType::trait>(cont); \
    }

#define DEFINE_PTR_BASED_CONST_BUFFER(func_name, trait)                                                       \
    template <typename T>                                                                                     \
    kamping::internal::PtrBasedConstBuffer<T, kamping::internal::ParameterType::trait> func_name(             \
        const T* ptr, size_t size) {                                                                          \
        return kamping::internal::PtrBasedConstBuffer<T, kamping::internal::ParameterType::trait>(ptr, size); \
    }


#define DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(func_name, parameter_trait, is_consumable) \
    template <typename Cont>                                                                \
    kamping::internal::UserAllocContainerBasedBuffer<                                       \
        Cont, kamping::internal::ParameterType::parameter_trait, is_consumable>             \
    func_name(Cont& cont) {                                                                 \
        return kamping::internal::UserAllocContainerBasedBuffer<                            \
            Cont, kamping::internal::ParameterType::parameter_trait, is_consumable>(cont);  \
    }
#define DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(func_name, parameter_trait, is_consumable) \
    template <typename T>                                                                    \
    kamping::internal::UserAllocUniquePtrBasedBuffer<                                        \
        T, kamping::internal::ParameterType::parameter_trait, is_consumable>                 \
    func_name(std::unique_ptr<T[]>& ptr) {                                                   \
        return kamping::internal::UserAllocUniquePtrBasedBuffer<                             \
            T, kamping::internal::ParameterType::parameter_trait, is_consumable>(ptr);       \
    }

#define DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER(func_name, parameter_trait)                                  \
    template <typename Cont>                                                                                 \
    kamping::internal::LibAllocContainerBasedBuffer<Cont, kamping::internal::ParameterType::parameter_trait> \
    func_name(NewContainer&&) {                                                                              \
        return kamping::internal::LibAllocContainerBasedBuffer<                                              \
            Cont, kamping::internal::ParameterType::parameter_trait>();                                      \
    }
#define DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER(func_name, parameter_trait)                                         \
    template <typename T>                                                                                            \
    kamping::internal::LibAllocUniquePtrBasedBuffer<T, kamping::internal::ParameterType::parameter_trait> func_name( \
        NewPtr&&) {                                                                                                  \
        return kamping::internal::LibAllocUniquePtrBasedBuffer<                                                      \
            T, kamping::internal::ParameterType::parameter_trait>();                                                 \
    }

#define DEFINE_MOVED_CONTAINER_BASED_BUFFER(func_name, parameter_trait, is_consumable)                     \
    template <typename Cont, typename = typename std::enable_if_t<!std::is_lvalue_reference<Cont>::value>> \
    kamping::internal::MovedContainerBasedBuffer<                                                          \
        Cont, kamping::internal::ParameterType::parameter_trait, is_consumable>                            \
    func_name(Cont&& cont) {                                                                               \
        return kamping::internal::MovedContainerBasedBuffer<                                               \
            Cont, kamping::internal::ParameterType::parameter_trait, is_consumable>(std::move(cont));      \
    }

} // namespace internal
DEFINE_CONTAINER_BASED_CONST_BUFFER(send_buf, send_buf)
DEFINE_PTR_BASED_CONST_BUFFER(send_buf, send_buf)

DEFINE_CONTAINER_BASED_CONST_BUFFER(send_counts, send_counts)
DEFINE_PTR_BASED_CONST_BUFFER(send_counts, send_counts)


DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(recv_buf, recv_buf, false)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_buf, recv_buf, false)
DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER(recv_buf, recv_buf)
DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_buf, recv_buf)
DEFINE_MOVED_CONTAINER_BASED_BUFFER(recv_buf, recv_buf, false);

DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(recv_counts, recv_counts, false)
DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(recv_counts_input, recv_counts, true)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_count, recv_counts, false)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_count_input, recv_counts, true)
DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER(recv_counts, recv_counts)
DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_counts, recv_counts)
DEFINE_MOVED_CONTAINER_BASED_BUFFER(recv_counts, recv_counts, false);
DEFINE_MOVED_CONTAINER_BASED_BUFFER(recv_counts_input, recv_counts, true);

DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(recv_displs, recv_displs, false)
DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(recv_displs_input, recv_displs, true)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_displs, recv_displs, false)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_displs_input, recv_displs, true)
DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER(recv_displs, recv_displs)
DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER(recv_displs, recv_displs)
DEFINE_MOVED_CONTAINER_BASED_BUFFER(recv_displs, recv_displs, false);
DEFINE_MOVED_CONTAINER_BASED_BUFFER(recv_displs_input, recv_displs, true);

DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(send_displs, send_displs, false)
DEFINE_USER_ALLOC_CONTAINER_BASED_BUFFER(send_displs_input, send_displs, true)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(send_displs, send_displs, false)
DEFINE_USER_ALLOC_UNIQUE_PTR_BASED_BUFFER(send_displs_input, send_displs, true)
DEFINE_LIB_ALLOC_CONTAINER_BASED_BUFFER(send_displs, send_displs)
DEFINE_LIB_ALLOC_UNIQUE_PTR_BASED_BUFFER(send_displs, send_displs)
DEFINE_MOVED_CONTAINER_BASED_BUFFER(send_displs, send_displs, false);
DEFINE_MOVED_CONTAINER_BASED_BUFFER(send_displs_input, send_displs, true);

/// @}

} // namespace kamping
