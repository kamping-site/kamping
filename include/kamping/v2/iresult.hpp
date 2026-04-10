#pragma once

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/result.hpp"

namespace kamping::v2 {

namespace detail {

/// Common base for all iresult specialisations.
///
/// Owns the MPI_Request, enforces move-only semantics, and provides:
///   - mpi_native_handle_ptr() for the MPI call site to fill in the request.
///   - do_wait() / do_test() — thin wrappers that call MPI_Wait/Test and throw
///     on error, so the derived wait()/test() methods only handle return values.
class iresult_base {
protected:
    MPI_Request request_ = MPI_REQUEST_NULL;

    void do_wait(MPI_Status* s) {
        int err = MPI_Wait(&request_, s);
        if (err != MPI_SUCCESS) {
            throw core::mpi_error(err);
        }
    }

    /// Returns true when the operation has completed.
    bool do_test(MPI_Status* s) {
        int flag;
        int err = MPI_Test(&request_, &flag, s);
        if (err != MPI_SUCCESS) {
            throw core::mpi_error(err);
        }
        return static_cast<bool>(flag);
    }

public:
    iresult_base() = default;

    iresult_base(iresult_base&&) noexcept            = default;
    iresult_base& operator=(iresult_base&&) noexcept = default;
    iresult_base(iresult_base const&)                = delete;
    iresult_base& operator=(iresult_base const&)     = delete;

    /// If the caller forgot to call wait()/test(), block on the outstanding request
    /// so the buffer is safe to destroy. Turns a leaked request into a visible stall
    /// rather than silent data-corruption. MPI errors are swallowed — a destructor
    /// cannot throw.
    ~iresult_base() {
        if (request_ != MPI_REQUEST_NULL) {
            (void)MPI_Wait(&request_, MPI_STATUS_IGNORE);
        }
    }

    MPI_Request* mpi_native_handle_ptr() {
        return &request_;
    }
};

} // namespace detail

// Primary template — specialised for one or two buffer types below.
template <typename... Bufs>
class iresult;

/// Single-buffer non-blocking result (isend, irecv).
///
/// Stores the buffer on the heap via a unique_ptr<view_t> so that the data pointer
/// MPI captures during MPI_Isend/Irecv remains valid regardless of whether the
/// iresult handle itself is moved. The MPI_Request is stored inline (it is a
/// copyable integer handle to a reference-counted MPI-internal object).
///
/// The view wrapping follows the usual all_t<Buf> convention:
///   - lvalue T&  → ref_view<T>:   non-owning; caller keeps the buffer alive.
///   - rvalue T   → owning_view<T>: owns the buffer until wait()/test() completes.
///   - view type  → stored as-is (all_t pass-through).
///
/// iresult is move-only: moving transfers ownership of the heap view (the view's
/// address is unchanged) and copies the MPI_Request handle.
template <typename Buf>
class iresult<Buf> : public detail::iresult_base {
    using view_t = ranges::all_t<Buf>;

    std::unique_ptr<view_t> view_;

    // Returns the buffer with correct ownership semantics:
    //   - lvalue Buf (T&):           ref_view<T> stored → base() → T&
    //   - non-view rvalue Buf (T):   owning_view<T> stored → move(*view_).base() → T&&
    //   - view rvalue Buf (View):    view stored as-is → move(*view_) → View&&
    decltype(auto) extract_buf() {
        if constexpr (std::is_lvalue_reference_v<Buf>) {
            return view_->base();
        } else if constexpr (std::derived_from<std::remove_cvref_t<Buf>, ranges::view_interface_base>) {
            return std::move(*view_);
        } else {
            return std::move(*view_).base();
        }
    }

public:
    explicit iresult(Buf&& buf)
        : view_(std::make_unique<view_t>(ranges::all(std::forward<Buf>(buf)))) {}

    view_t& view() {
        return *view_;
    }

    view_t const& view() const {
        return *view_;
    }

    // ── Completion ───────────────────────────────────────────────────────────

    /// Blocks until the operation completes, then returns the buffer.
    ///   owned  (owning_view<T>): moves value out — iresult is left in moved-from state.
    ///   borrowed (ref_view<T>):  returns lvalue reference to the external buffer.
    ///   view pass-through:       moves the view out.
    template <bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    decltype(auto) wait(Status&& status = MPI_STATUS_IGNORE) {
        do_wait(kamping::bridge::native_handle_ptr(status));
        return extract_buf();
    }

    /// Non-blocking completion check.
    ///   borrowed (ref_view<T> or borrowed view): returns bool.
    ///   owned (owning_view<T> or non-borrowed view): returns optional<T> / optional<View>.
    ///     Some on completion — buffer moved out, iresult spent.
    ///     nullopt on pending — iresult remains valid for retry or wait().
    template <bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    auto test(Status&& status = MPI_STATUS_IGNORE) {
        bool const done = do_test(kamping::bridge::native_handle_ptr(status));
        if constexpr (ranges::borrowed_buffer<view_t>) {
            return done;
        } else {
            using value_t = std::remove_reference_t<decltype(extract_buf())>;
            if (done)
                return std::optional<value_t>{extract_buf()};
            return std::optional<value_t>{std::nullopt};
        }
    }
};

/// Two-buffer non-blocking result (non-blocking sendrecv, non-blocking collectives).
///
/// Stores a result<SBuf, RBuf> on the heap for pointer stability. The MPI_Request
/// is stored inline (copyable integer handle). Mirrors iresult<Buf>:
///   - lvalue SBuf/RBuf: reference members — caller keeps the buffers alive.
///   - rvalue SBuf/RBuf: value members — iresult owns the buffers until wait()/test().
/// Move-only: moving transfers the unique_ptr without relocating the buffers.
template <typename SBuf, typename RBuf>
class iresult<SBuf, RBuf> : public detail::iresult_base {
    std::unique_ptr<result<SBuf, RBuf>> result_;

public:
    explicit iresult(SBuf&& sbuf, RBuf&& rbuf)
        : result_(std::make_unique<result<SBuf, RBuf>>(
              result<SBuf, RBuf>{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)})) {}

    // ── Buffer access ─────────────────────────────────────────────────────────

    SBuf& send() {
        return result_->send;
    }
    RBuf& recv() {
        return result_->recv;
    }

    // ── Completion ────────────────────────────────────────────────────────────

    /// Blocks until complete, then returns the result.
    /// Owned members are moved out; borrowed members copy their reference binding (no data copy).
    template <bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    result<SBuf, RBuf> wait(Status&& status = MPI_STATUS_IGNORE) {
        do_wait(kamping::bridge::native_handle_ptr(status));
        return std::move(*result_);
    }

    /// Non-blocking completion check. Always returns optional<result<SBuf, RBuf>>.
    ///   Some: operation complete — owned members moved out, ref bindings copied (no data copy).
    ///   nullopt: pending — iresult remains valid for retry or wait().
    template <bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
    std::optional<result<SBuf, RBuf>> test(Status&& status = MPI_STATUS_IGNORE) {
        if (do_test(kamping::bridge::native_handle_ptr(status)))
            return std::move(*result_);
        return std::nullopt;
    }
};

} // namespace kamping::v2
