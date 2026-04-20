#pragma once

#include <string_view>
#include <utility>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/group.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

// ── CRTP mixin ──────────────────────────────────────────────────────────────
// Provides read-only accessors for any communicator wrapper:
// `.rank()`, `.size()`, `.native()`, `.group()`.
// Derived must implement `mpi_handle() const → MPI_Comm`.

template <typename Derived>
class comm_accessors {
    [[nodiscard]] MPI_Comm underlying() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @return The rank of the calling process in this communicator.
    [[nodiscard]] int rank() const {
        int r;
        MPI_Comm_rank(underlying(), &r);
        return r;
    }

    /// @return The number of processes in this communicator.
    [[nodiscard]] int size() const {
        int s;
        MPI_Comm_size(underlying(), &s);
        return s;
    }

    /// @brief Extract the group associated with this communicator.
    ///
    /// The returned `group` is owned by the caller.
    /// @throws mpi_error if `MPI_Comm_group` fails.
    [[nodiscard]] group group() const {
        MPI_Group g = MPI_GROUP_EMPTY;
        int       err = MPI_Comm_group(underlying(), &g);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return mpi::experimental::group::from_native(g);
    }

    /// @return The underlying `MPI_Comm` (escape hatch).
    [[nodiscard]] MPI_Comm native() const noexcept { return underlying(); }
};

// ── comm_view ────────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Comm`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Comm>`. Does not free the
/// communicator on destruction — use the owning `comm` for that.
/// Provides read-only accessors only; use the free functions `dup()` /
/// `split()` or the owning `comm` methods to create sub-communicators.
class comm_view : public comm_accessors<comm_view> {
public:
    /// @brief Construct from a raw `MPI_Comm`. The communicator must outlive this view.
    explicit comm_view(MPI_Comm c) noexcept : _comm(c) {}

    /// @return The underlying `MPI_Comm` (for `handle()` dispatch).
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept { return _comm; }

private:
    MPI_Comm _comm;
};

// ── comm ─────────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Comm`.
///
/// Move-only. Calls `MPI_Comm_free` on destruction unless the stored handle
/// is `MPI_COMM_NULL`.
///
/// **Invariant**: only communicators created by the caller (via `MPI_Comm_dup`,
/// `MPI_Comm_split`, `MPI_Comm_create_from_group`, etc.) should be passed to
/// `from_native`. Never adopt `MPI_COMM_WORLD` or `MPI_COMM_SELF`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Comm>`.
class comm : public comm_accessors<comm> {
public:
    /// @brief Default-construct a null communicator (holds `MPI_COMM_NULL`).
    ///
    /// Useful as an output parameter for the free functions `dup()` / `split()`:
    /// @code
    ///   comm out;
    ///   mpi::experimental::dup(source, out);
    /// @endcode
    comm() noexcept = default;

    comm(comm const&)            = delete;
    comm& operator=(comm const&) = delete;

    /// @brief Move constructor — transfers ownership; moved-from holds `MPI_COMM_NULL`.
    comm(comm&& o) noexcept : _comm(std::exchange(o._comm, MPI_COMM_NULL)) {}

    /// @brief Move assignment — frees the current handle then transfers ownership.
    comm& operator=(comm&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _comm = std::exchange(o._comm, MPI_COMM_NULL);
        }
        return *this;
    }

    /// @brief Free the communicator (unless moved from or null).
    ~comm() noexcept {
        free_if_valid();
    }

    /// @brief Adopt an already-created `MPI_Comm` (takes ownership).
    ///
    /// Do not pass predefined communicators (`MPI_COMM_WORLD`, `MPI_COMM_SELF`).
    [[nodiscard]] static comm from_native(MPI_Comm c) noexcept {
        return comm(c, adopt_t{});
    }

    /// @brief Collective: create a new communicator from a group.
    ///
    /// Calls `MPI_Comm_create_from_group`. All processes in `g` must call this
    /// collectively with the same `tag`.
    ///
    /// @param g    Group defining the new communicator's process set.
    /// @param tag  Application-defined string tag (must match across all callers).
    /// @param info MPI info hints (`MPI_INFO_NULL` for defaults).
    /// @throws mpi_error if the MPI call fails.
    explicit comm(
        mpi::experimental::group const& g,
        std::string_view                tag  = "",
        MPI_Info                        info = MPI_INFO_NULL
    ) {
        MPI_Comm c   = MPI_COMM_NULL;
        int      err = MPI_Comm_create_from_group(g.mpi_handle(), tag.data(), info, MPI_ERRORS_RETURN, &c);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        _comm = c;
    }

    /// @brief Collective: duplicate this communicator (MPI_Comm_dup).
    ///
    /// All processes in the communicator must call this collectively.
    /// Delegates to the free function `mpi::experimental::dup(*this)`.
    /// @throws mpi_error if `MPI_Comm_dup` fails.
    [[nodiscard]] comm dup() const;

    /// @brief Collective: split this communicator by color and key (MPI_Comm_split).
    ///
    /// Processes with the same `color` form a sub-communicator ordered by `key`.
    /// Pass `MPI_UNDEFINED` as `color` to opt out — result `.native() == MPI_COMM_NULL`.
    /// Delegates to the free function `mpi::experimental::split(*this, color, key)`.
    /// @throws mpi_error if `MPI_Comm_split` fails.
    [[nodiscard]] comm split(int color, int key = 0) const;

    /// @brief Implicit conversion to a non-owning view.
    ///
    /// Allows passing a `comm` wherever a `comm_view` is expected without
    /// an explicit cast. The view borrows; it does not extend lifetime.
    operator comm_view() const noexcept { return comm_view{_comm}; }

    /// @return The underlying `MPI_Comm` (for `handle()` dispatch).
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept { return _comm; }

    /// @brief Pointer to the underlying `MPI_Comm`, for use as an MPI out-parameter.
    ///
    /// **Precondition**: the comm must be null (`MPI_COMM_NULL`). Writing into a
    /// non-null `comm` bypasses `MPI_Comm_free` and leaks the existing handle.
    MPI_Comm* mpi_handle_ptr() noexcept { return &_comm; }

private:
    struct adopt_t {};

    explicit comm(MPI_Comm c, adopt_t) noexcept : _comm(c) {}

    void free_if_valid() noexcept {
        if (_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&_comm);
            _comm = MPI_COMM_NULL;
        }
    }

    MPI_Comm _comm = MPI_COMM_NULL;
};

// ── Free functions: dup / split ───────────────────────────────────────────────
// Mirror the C API closely: one input argument (convertible to MPI_Comm value)
// and one output argument (convertible to MPI_Comm*). Return void — the caller
// decides the type of the output handle. The owning `comm` member methods layer
// RAII on top by passing a local MPI_Comm and adopting the result.

/// @brief Collective: duplicate a communicator (MPI_Comm_dup).
///
/// @tparam Comm     Any type satisfying `convertible_to_mpi_handle<MPI_Comm>`.
/// @tparam NewComm  Any type satisfying `convertible_to_mpi_handle_ptr<MPI_Comm>`.
/// @param c        Input communicator.
/// @param new_comm Output: receives the duplicated communicator handle.
/// @throws mpi_error if `MPI_Comm_dup` fails.
template <
    convertible_to_mpi_handle<MPI_Comm>     Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Comm> NewComm = MPI_Comm*>
void dup(Comm const& c, NewComm& new_comm) {
    int err = MPI_Comm_dup(handle(c), handle_ptr(new_comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

/// @brief Collective: split a communicator by color and key (MPI_Comm_split).
///
/// Processes with the same `color` form a sub-communicator, ordered by `key`.
/// Pass `MPI_UNDEFINED` as `color` to opt out — the output handle is set to
/// `MPI_COMM_NULL` for that process.
///
/// @tparam Comm     Any type satisfying `convertible_to_mpi_handle<MPI_Comm>`.
/// @tparam NewComm  Any type satisfying `convertible_to_mpi_handle_ptr<MPI_Comm>`.
/// @param c        Input communicator.
/// @param new_comm Output: receives the new sub-communicator handle.
/// @throws mpi_error if `MPI_Comm_split` fails.
template <
    convertible_to_mpi_handle<MPI_Comm>     Comm    = MPI_Comm,
    convertible_to_mpi_handle_ptr<MPI_Comm> NewComm = MPI_Comm*>
void split(Comm const& c, int color, int key, NewComm& new_comm) {
    int err = MPI_Comm_split(handle(c), color, key, handle_ptr(new_comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

// ── comm member definitions (out-of-line; layer RAII on top of free functions) ─

inline comm comm::dup() const {
    MPI_Comm out = MPI_COMM_NULL;
    mpi::experimental::dup(*this, out);
    return comm::from_native(out);
}

inline comm comm::split(int color, int key) const {
    MPI_Comm out = MPI_COMM_NULL;
    mpi::experimental::split(*this, color, key, out);
    return comm::from_native(out);
}

} // namespace mpi::experimental
