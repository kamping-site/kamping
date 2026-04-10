#pragma once

#include <mpi.h>

#include "kamping/builtin_types.hpp"

namespace kamping::v2 {

// ── CRTP mixin ──────────────────────────────────────────────────────────────
// Provides the shared read accessors (.source(), .tag(), .count<T>()) for both
// status and status_view.  Derived must implement `status_ptr() const -> MPI_Status const*`.

template <typename Derived>
class status_accessors {
    [[nodiscard]] MPI_Status const* ptr() const {
        return static_cast<Derived const*>(this)->status_ptr();
    }

public:
    /// @return The source rank.
    [[nodiscard]] int source() const {
        return ptr()->MPI_SOURCE;
    }

    /// @return The tag.
    [[nodiscard]] int tag() const {
        return ptr()->MPI_TAG;
    }

    /// @param datatype The MPI datatype.
    /// @return The number of top-level elements of the given datatype in the message.
    [[nodiscard]] int count(MPI_Datatype datatype) const {
        int cnt;
        MPI_Get_count(ptr(), datatype, &cnt);
        return cnt;
    }

    /// @tparam T A builtin MPI-mappable type.
    /// @return The number of top-level elements of type `T` in the message.
    template <typename T>
        requires kamping::is_builtin_type_v<T>
    [[nodiscard]] int count() const {
        return count(kamping::builtin_type<T>::data_type());
    }
};

// ── status_view ─────────────────────────────────────────────────────────────

/// @brief Non-owning view over an existing `MPI_Status`.
///
/// Satisfies `bridge::convertible_to_mpi_handle_ptr<MPI_Status>` so it can be
/// passed directly to `core::` operations that accept a status out-parameter.
class status_view : public status_accessors<status_view> {
public:
    /// @brief Construct from a pointer to an existing MPI_Status. Must not be null.
    explicit status_view(MPI_Status* status) : _status(status) {}

    /// @return Pointer to the underlying `MPI_Status` (escape hatch).
    [[nodiscard]] MPI_Status* native() const {
        return _status;
    }

    // ── bridge customization points ──────────────────────────────────────────

    /// @return The underlying `MPI_Status` by value (for `bridge::native_handle`).
    [[nodiscard]] MPI_Status mpi_native_handle() const {
        return *_status;
    }

    /// @return Pointer to the underlying `MPI_Status` (for `bridge::native_handle_ptr`).
    [[nodiscard]] MPI_Status* mpi_native_handle_ptr() {
        return _status;
    }

private:
    friend class status_accessors<status_view>;
    [[nodiscard]] MPI_Status const* status_ptr() const {
        return _status;
    }

    MPI_Status* _status; ///< Non-owning pointer to the viewed status.
};

// ── status ───────────────────────────────────────────────────────────────────

/// @brief Owning wrapper around `MPI_Status`.
///
/// All fields are undefined until the object is passed to (or filled by) an MPI
/// communication function.  Satisfies both `bridge::convertible_to_mpi_handle<MPI_Status>`
/// and `bridge::convertible_to_mpi_handle_ptr<MPI_Status>`.
class status : public status_accessors<status> {
public:
    /// @brief Default-construct. All fields are undefined until filled by MPI.
    status() : _status() {}

    /// @brief Construct from an existing `MPI_Status`.
    explicit status(MPI_Status s) : _status(s) {}

    /// @return Reference to the underlying `MPI_Status` (escape hatch).
    [[nodiscard]] MPI_Status& native() {
        return _status;
    }

    /// @return Const reference to the underlying `MPI_Status` (escape hatch).
    [[nodiscard]] MPI_Status const& native() const {
        return _status;
    }

    // ── bridge customization points ──────────────────────────────────────────

    /// @return The underlying `MPI_Status` by value (for `bridge::native_handle`).
    [[nodiscard]] MPI_Status mpi_native_handle() const {
        return _status;
    }

    /// @return Pointer to the underlying `MPI_Status` (for `bridge::native_handle_ptr`).
    [[nodiscard]] MPI_Status* mpi_native_handle_ptr() {
        return &_status;
    }

private:
    friend class status_accessors<status>;
    [[nodiscard]] MPI_Status const* status_ptr() const {
        return &_status;
    }

    MPI_Status _status; ///< The owned status.
};

} // namespace kamping::v2
