#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include <cereal/archives/binary.hpp>

#include "kamping/builtin_types.hpp"
#include "kamping/v2/ranges/adaptor_closure.hpp"

namespace kamping::ranges {

/// Wraps an object and serializes/deserializes it with cereal for MPI transport.
///
/// T is the wrapped type: a (possibly const) lvalue reference for non-owning views, or
/// a value type for owning views. Use `obj | kamping::views::serialize` to construct.
///
/// Send path: mpi_size()/mpi_data() lazily serialize the wrapped object into buffer_ on first
///            access; the ostringstream result is moved (no copy) into buffer_.
/// Recv path: set_recv_count(n) sizes buffer_ for MPI to write into directly; operator*
///            triggers lazy deserialization via a zero-copy membuf streambuf, then
///            clears buffer_ to leave a deterministic empty state. Requires non-const T.
///
/// Range semantics are intentionally omitted. Access the wrapped object via operator* or
/// operator->; for ranges, dereference first: `for (auto& x : *view) { ... }`.
template <typename T, typename Alloc = std::allocator<char>>
class serialization_view {
    static constexpr bool is_owning = !std::is_lvalue_reference_v<T>;
    using value_type                = std::remove_reference_t<T>;
    // Lvalue-ref case: store a (possibly const) pointer to avoid requiring copyability.
    // Owning case: store by value.
    using stored_t = std::conditional_t<is_owning, value_type, value_type*>;

    mutable stored_t base_;

    mutable std::basic_string<char, std::char_traits<char>, Alloc> buffer_;
    mutable bool   serialized_            = false;
    mutable bool   needs_deserialization_ = false;
    std::ptrdiff_t recv_count_            = 0;

    // base_ is mutable, so this is safe from const methods.
    // The const-lvalue-ref case (value_type is const-qualified) is prevented from
    // reaching do_deserialize() by the set_recv_count requires-clause.
    value_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return base_;
        else
            return *base_;
    }

    void do_serialize() const {
        std::basic_ostringstream<char> oss;
        {
            cereal::BinaryOutputArchive ar(oss);
            ar(base_ref());
        }
        buffer_     = std::move(oss).str(); // move — no copy
        serialized_ = true;
    }

    // Zero-copy read-only streambuf over an existing char buffer.
    // std::basic_ispanstream (C++23) would be cleaner but isn't universally available yet.
    struct membuf : std::basic_streambuf<char> {
        membuf(char const* data, std::size_t size) {
            auto p = const_cast<char*>(data); // setg requires non-const; we only read
            setg(p, p, p + size);
        }
    };

    void do_deserialize() const {
        membuf                   mb(buffer_.data(), buffer_.size());
        std::basic_istream<char> is(&mb);
        cereal::BinaryInputArchive ar(is);
        ar(base_ref());
        buffer_.clear(); // received bytes no longer needed; known empty state
        needs_deserialization_ = false;
    }

public:
    /// Non-owning constructor: stores a pointer to the referenced object.
    /// Handles both `T&` and `T const&` (value_type may be const-qualified).
    explicit serialization_view(value_type& obj) requires(!is_owning) : base_(&obj) {}

    /// Owning constructor: takes ownership of a moved object.
    explicit serialization_view(value_type&& obj) requires(is_owning) : base_(std::move(obj)) {}

    /// Dereference to the wrapped object, triggering deserialization if needed.
    value_type& operator*() {
        if (needs_deserialization_) do_deserialize();
        return base_ref();
    }

    value_type const& operator*() const {
        if (needs_deserialization_) do_deserialize();
        return base_ref();
    }

    value_type*       operator->() { return std::addressof(**this); }
    value_type const* operator->() const { return std::addressof(**this); }

    // ---- Recv-side protocol -----------------------------------------------

    /// Called by infer() with the number of bytes to receive. Requires non-const T:
    /// deserialization writes into the wrapped object.
    void set_recv_count(std::ptrdiff_t n) requires(!std::is_const_v<value_type>) {
        recv_count_           = n;
        serialized_           = false;
        needs_deserialization_ = true;
        buffer_.resize(static_cast<std::size_t>(n));
    }

    // ---- MPI protocol methods --------------------------------------------

    std::ptrdiff_t mpi_size() const {
        if (needs_deserialization_) return recv_count_;
        if (!serialized_) do_serialize();
        return static_cast<std::ptrdiff_t>(buffer_.size());
    }

    MPI_Datatype mpi_type() const {
        return kamping::builtin_type<char>::data_type();
    }

    /// Returns a mutable pointer: satisfies send_buffer (void const* accepted) and
    /// recv_buffer (void* required). Serializes lazily on the send side.
    void* mpi_data() const {
        if (!needs_deserialization_ && !serialized_) do_serialize();
        return buffer_.data();
    }
};

// lvalue input (including const lvalue): non-owning.
// T deduced as U or U const → serialization_view<U&> or serialization_view<U const&>.
template <typename T>
serialization_view(T&) -> serialization_view<T&>;

// rvalue input: owning.
template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
serialization_view(T&&) -> serialization_view<T>;


} // namespace kamping::ranges

namespace kamping::views {
inline constexpr struct serialize_fn : kamping::ranges::adaptor_closure<serialize_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::ranges::serialization_view(std::forward<R>(r));
    }
} serialize{};

/// Returns an owning serialization_view<T> with a default-constructed T.
/// Use as a recv buffer when the object does not exist yet:
///   auto view = comm.recv(kamping::views::deserialize<MyType>(), 0);
///   MyType& result = *view;
template <typename T>
    requires std::default_initializable<T>
auto deserialize() {
    return kamping::ranges::serialization_view<T>(T{});
}
} // namespace kamping::views
